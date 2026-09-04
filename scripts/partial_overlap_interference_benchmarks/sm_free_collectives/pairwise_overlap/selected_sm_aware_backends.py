# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""NCCL adapters for the SM-free pairwise overlap matrix.

All-gather and balanced A2A use NCCL 2.28.3's existing Zero-CTA collective
implementations.  Their buffers come from ``ncclMemAlloc``, are registered as
``CollSymmetric`` windows, and the communicator is constructed with
``CTAPolicy.Zero``.  Setup preloads the repository-local NCCL/CUDA-compat
overlay before Torch; version, driver API, allocator, window registration, and
policy checks all fail closed.

This module deliberately imports neither Torch nor Ray at import time so the
driver can resolve, hash, and validate selectors without creating a CUDA
context or reserving GPUs.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import sys
from typing import Any, Mapping


ALL_GATHER_SELECTOR = (
    "scripts.partial_overlap_interference_benchmarks.sm_free_collectives."
    "pairwise_overlap.selected_sm_aware_backends:"
    "create_all_gather_nccl_228_zero_cta"
)
BALANCED_A2A_SELECTOR = (
    "scripts.partial_overlap_interference_benchmarks.sm_free_collectives."
    "pairwise_overlap.selected_sm_aware_backends:"
    "create_balanced_a2a_nccl_228_zero_cta"
)
# Compatibility alias for intra-node A2A control experiments.
RAGGED_A2A_SELECTOR = BALANCED_A2A_SELECTOR
ALL_GATHER_NORMAL_SELECTOR = (
    "scripts.partial_overlap_interference_benchmarks.sm_free_collectives."
    "pairwise_overlap.selected_sm_aware_backends:"
    "create_all_gather_torch_distributed_nccl_default_cta"
)
BALANCED_A2A_NORMAL_SELECTOR = (
    "scripts.partial_overlap_interference_benchmarks.sm_free_collectives."
    "pairwise_overlap.selected_sm_aware_backends:"
    "create_balanced_a2a_torch_distributed_nccl_default_cta"
)
MULTIMEM_ALL_GATHER_SELECTOR = (
    "scripts.partial_overlap_interference_benchmarks.sm_free_collectives."
    "pairwise_overlap.selected_sm_aware_backends:"
    "create_all_gather_torch_symm_mem_multimem"
)

_COMMON_COPY_OPTIONS = frozenset({"cuda_runtime_library", "cuda_driver_library"})
_ZERO_CTA_OPTIONS = frozenset({
    "cuda_compat_path",
    "overlay_dir",
    "expected_nccl_version",
})
_MULTIMEM_OPTIONS = frozenset({"use_physical_cuda_ordinals"})

_REPOSITORY_ZERO_CTA_OVERLAY = (
    Path(__file__).resolve().parent / ".runtime/nccl_2_28_overlay"
)
_ZERO_CTA_FAMILIES = frozenset({"all_gather", "balanced_a2a"})
_PREPARED_ZERO_CTA: tuple[Path, str, Path] | None = None


def _options(
    family: str,
    raw: Mapping[str, Any],
    allowed: frozenset[str],
) -> dict[str, Any]:
    result = dict(raw)
    unknown = sorted(set(result) - set(allowed))
    if unknown:
        raise ValueError(f"unsupported {family} backend options: {unknown}")
    return result


def _require_context(
    context: Mapping[str, Any],
    *,
    family: str,
    group_size: int,
    names: tuple[str, ...],
) -> None:
    if context.get("family") != family:
        raise ValueError(
            f"{family} factory received family={context.get('family')!r}"
        )
    missing = sorted(name for name in names if name not in context)
    if missing:
        raise ValueError(f"{family} context is missing {missing}")
    if int(context["group_size"]) != group_size:
        raise ValueError(
            f"{family} requires group_size={group_size}, got "
            f"{context['group_size']!r}"
        )
    group_rank = int(context["group_rank"])
    if not 0 <= group_rank < group_size:
        raise ValueError(f"invalid {family} group rank {group_rank}")


def _zero_cta_settings(
    family: str, options: Mapping[str, Any]
) -> tuple[Path, str, Path]:
    if family not in _ZERO_CTA_FAMILIES:
        raise ValueError(f"{family} is not a Zero-CTA collective family")
    parsed = _options(family, options, _ZERO_CTA_OPTIONS)
    overlay = Path(parsed.get(
        "overlay_dir",
        os.environ.get("PIPER_ZERO_CTA_OVERLAY", _REPOSITORY_ZERO_CTA_OVERLAY),
    )).resolve()
    expected = str(parsed.get(
        "expected_nccl_version",
        os.environ.get("EXPECTED_NCCL_VERSION", "2.28.3"),
    ))
    if expected != "2.28.3":
        raise ValueError(
            f"the selected Zero-CTA backends require NCCL 2.28.3, got {expected}"
        )
    default_container_compat = Path("/usr/local/cuda-12.8/compat")
    bundled_compat = (
        overlay / "cuda_compat/usr/local/cuda-12.5/compat"
    )
    compat_path = Path(parsed.get(
        "cuda_compat_path",
        os.environ.get(
            "CUDA_COMPAT_PATH",
            (
                default_container_compat
                if (default_container_compat / "libcuda.so.1").is_file()
                else bundled_compat
            ),
        ),
    )).resolve()
    required = (
        overlay / "manifest.json",
        overlay / "python/nvidia/nccl/lib/libnccl.so.2",
        compat_path / "libcuda.so.1",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete NCCL 2.28.3 overlay: {missing}")
    return overlay, expected, compat_path


def _prepend_path(value: str, additions: list[str]) -> str:
    result: list[str] = []
    for item in [*additions, *value.split(os.pathsep)]:
        if item and item not in result:
            result.append(item)
    return os.pathsep.join(result)


def worker_environment(
    *, family: str, options: Mapping[str, Any], environment: Mapping[str, str]
) -> dict[str, str]:
    """Give Ray actors the loader paths needed before their process starts."""

    if family == "all_gather" and options.get("use_physical_cuda_ordinals"):
        parsed = _options(family, options, _MULTIMEM_OPTIONS)
        if parsed.get("use_physical_cuda_ordinals") is not True:
            raise ValueError("use_physical_cuda_ordinals must be true")
        return {
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
            "PIPER_USE_PHYSICAL_CUDA_ORDINAL": "1",
        }
    if family not in _ZERO_CTA_FAMILIES:
        return {}
    overlay, expected, compat_path = _zero_cta_settings(family, options)
    compat = str(compat_path)
    nccl_lib = str(overlay / "python/nvidia/nccl/lib")
    python_dir = str(overlay / "python")
    return {
        "CUDA_COMPAT_PATH": compat,
        "EXPECTED_NCCL_VERSION": expected,
        "NCCL4PY_OVERLAY": python_dir,
        "PIPER_ZERO_CTA_OVERLAY": str(overlay),
        "LD_LIBRARY_PATH": _prepend_path(
            str(environment.get("LD_LIBRARY_PATH", "")), [compat, nccl_lib]
        ),
        "PYTHONPATH": _prepend_path(
            str(environment.get("PYTHONPATH", "")), [python_dir]
        ),
    }


def _mapped_libraries(fragment: str) -> list[str]:
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        return []
    return sorted({
        line.split()[-1]
        for line in maps.read_text(encoding="utf-8").splitlines()
        if fragment in line and line.split()[-1].startswith("/")
    })


def _version_code(version: str) -> int:
    major, minor, patch = (int(value) for value in version.split("."))
    return major * 10_000 + minor * 100 + patch


def prepare_worker_process(*, family: str, options: Mapping[str, Any]) -> None:
    """Preload CUDA compatibility and NCCL before Torch maps another NCCL."""

    global _PREPARED_ZERO_CTA
    if family == "all_gather" and options.get("use_physical_cuda_ordinals"):
        parsed = _options(family, options, _MULTIMEM_OPTIONS)
        if parsed.get("use_physical_cuda_ordinals") is not True:
            raise ValueError("use_physical_cuda_ordinals must be true")
        return
    if family not in _ZERO_CTA_FAMILIES:
        return
    overlay, expected, compat_dir = _zero_cta_settings(family, options)
    identity = (overlay, expected, compat_dir)
    if _PREPARED_ZERO_CTA is not None:
        if _PREPARED_ZERO_CTA != identity:
            raise RuntimeError(
                "Zero-CTA families requested different runtime overlays: "
                f"{_PREPARED_ZERO_CTA} != {identity}"
            )
        return

    expected_nccl = (
        overlay / "python/nvidia/nccl/lib/libnccl.so.2"
    ).resolve()
    mapped_nccl = _mapped_libraries("libnccl.so")
    if mapped_nccl and str(expected_nccl) not in mapped_nccl:
        raise RuntimeError(
            "libnccl was mapped before the NCCL 2.28.3 overlay preload: "
            f"{mapped_nccl}"
        )

    compat_driver = ctypes.CDLL(
        str(compat_dir / "libcuda.so.1"), mode=ctypes.RTLD_GLOBAL
    )
    compat_driver.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
    compat_driver.cuDriverGetVersion.restype = ctypes.c_int
    driver_version = ctypes.c_int()
    status = int(compat_driver.cuDriverGetVersion(ctypes.byref(driver_version)))
    if status != 0 or int(driver_version.value) < 12_050:
        raise RuntimeError(
            "Zero-CTA requires CUDA Driver API >= 12.5: "
            f"status={status}, version={driver_version.value}"
        )
    compat_driver.cuInit.argtypes = [ctypes.c_uint]
    compat_driver.cuInit.restype = ctypes.c_int
    init_status = int(compat_driver.cuInit(0))
    if init_status != 0:
        raise RuntimeError(f"CUDA compatibility cuInit failed with {init_status}")

    nccl_library = ctypes.CDLL(str(expected_nccl), mode=ctypes.RTLD_GLOBAL)
    nccl_library.ncclGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
    nccl_library.ncclGetVersion.restype = ctypes.c_int
    nccl_version = ctypes.c_int()
    status = int(nccl_library.ncclGetVersion(ctypes.byref(nccl_version)))
    if status != 0 or int(nccl_version.value) != _version_code(expected):
        raise RuntimeError(
            f"expected NCCL {expected}, ncclGetVersion returned "
            f"status={status}, version={nccl_version.value}"
        )
    python_dir = str(overlay / "python")
    if python_dir not in sys.path:
        sys.path.insert(0, python_dir)
    _PREPARED_ZERO_CTA = identity


class _CudaIpcMemHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * 64)]


class _CudaRuntime:
    """Small CUDA runtime/driver binding for cache-allocation IPC export."""

    def __init__(
        self,
        runtime_library: str = "libcudart.so.12",
        driver_library: str = "libcuda.so.1",
    ) -> None:
        self.runtime = ctypes.CDLL(runtime_library)
        self.driver = ctypes.CDLL(driver_library)
        self._bind_runtime()
        self._bind_driver()
        self._driver_check(self.driver.cuInit(0), "cuInit")

    def _bind_runtime(self) -> None:
        runtime = self.runtime
        runtime.cudaGetErrorString.argtypes = [ctypes.c_int]
        runtime.cudaGetErrorString.restype = ctypes.c_char_p
        runtime.cudaIpcGetMemHandle.argtypes = [
            ctypes.POINTER(_CudaIpcMemHandle), ctypes.c_void_p
        ]
        runtime.cudaIpcGetMemHandle.restype = ctypes.c_int
        runtime.cudaIpcOpenMemHandle.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), _CudaIpcMemHandle, ctypes.c_uint
        ]
        runtime.cudaIpcOpenMemHandle.restype = ctypes.c_int
        runtime.cudaIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
        runtime.cudaIpcCloseMemHandle.restype = ctypes.c_int
        runtime.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        runtime.cudaMemcpyAsync.restype = ctypes.c_int

    def _bind_driver(self) -> None:
        driver = self.driver
        driver.cuInit.argtypes = [ctypes.c_uint]
        driver.cuInit.restype = ctypes.c_int
        driver.cuGetErrorString.argtypes = [
            ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)
        ]
        driver.cuGetErrorString.restype = ctypes.c_int
        address_range = getattr(driver, "cuMemGetAddressRange_v2", None)
        if address_range is None:
            address_range = driver.cuMemGetAddressRange
        address_range.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_uint64,
        ]
        address_range.restype = ctypes.c_int
        self._address_range = address_range

    def _runtime_check(self, status: int, operation: str) -> None:
        if status == 0:
            return
        raw = self.runtime.cudaGetErrorString(status)
        detail = "" if raw is None else raw.decode("utf-8", errors="replace")
        raise RuntimeError(f"{operation} failed: CUDA runtime {status}: {detail}")

    def _driver_check(self, status: int, operation: str) -> None:
        if status == 0:
            return
        raw = ctypes.c_char_p()
        self.driver.cuGetErrorString(status, ctypes.byref(raw))
        detail = "" if raw.value is None else raw.value.decode(errors="replace")
        raise RuntimeError(f"{operation} failed: CUDA driver {status}: {detail}")

    def export_tensor(self, pointer: int, nbytes: int) -> dict[str, Any]:
        if pointer <= 0 or nbytes <= 0:
            raise ValueError("CUDA IPC export requires a nonempty device tensor")
        base = ctypes.c_uint64()
        allocation_bytes = ctypes.c_size_t()
        self._driver_check(
            self._address_range(
                ctypes.byref(base), ctypes.byref(allocation_bytes),
                ctypes.c_uint64(pointer),
            ),
            "cuMemGetAddressRange",
        )
        offset = pointer - int(base.value)
        if offset < 0 or offset + nbytes > int(allocation_bytes.value):
            raise RuntimeError(
                "tensor range is outside its CUDA allocation: "
                f"offset={offset}, bytes={nbytes}, allocation={allocation_bytes.value}"
            )
        handle = _CudaIpcMemHandle()
        self._runtime_check(
            self.runtime.cudaIpcGetMemHandle(
                ctypes.byref(handle), ctypes.c_void_p(base.value)
            ),
            "cudaIpcGetMemHandle",
        )
        raw = ctypes.string_at(ctypes.byref(handle), ctypes.sizeof(handle))
        return {
            "handle_hex": raw.hex(),
            "allocation_bytes": int(allocation_bytes.value),
            "tensor_offset_bytes": int(offset),
            "tensor_nbytes": int(nbytes),
        }

    def open_tensor(self, descriptor: Mapping[str, Any]) -> tuple[int, int]:
        raw = bytes.fromhex(str(descriptor["handle_hex"]))
        if len(raw) != ctypes.sizeof(_CudaIpcMemHandle):
            raise ValueError("CUDA IPC memory handle must contain exactly 64 bytes")
        handle = _CudaIpcMemHandle()
        ctypes.memmove(ctypes.byref(handle), raw, len(raw))
        opened = ctypes.c_void_p()
        self._runtime_check(
            self.runtime.cudaIpcOpenMemHandle(
                ctypes.byref(opened), handle, 1  # cudaIpcMemLazyEnablePeerAccess
            ),
            "cudaIpcOpenMemHandle",
        )
        base = int(opened.value or 0)
        if not base:
            raise RuntimeError("cudaIpcOpenMemHandle returned a null pointer")
        return base, base + int(descriptor["tensor_offset_bytes"])

    def close_mapping(self, base_pointer: int) -> None:
        self._runtime_check(
            self.runtime.cudaIpcCloseMemHandle(ctypes.c_void_p(base_pointer)),
            "cudaIpcCloseMemHandle",
        )

    def copy_async(
        self,
        *,
        destination: int,
        source: int,
        nbytes: int,
        stream: int,
    ) -> None:
        if nbytes <= 0:
            return
        self._runtime_check(
            self.runtime.cudaMemcpyAsync(
                ctypes.c_void_p(destination),
                ctypes.c_void_p(source),
                ctypes.c_size_t(nbytes),
                4,  # cudaMemcpyDefault: infer peer route from CUDA IPC/UVA pointers.
                ctypes.c_void_p(stream),
            ),
            "cudaMemcpyAsync(CUDA IPC peer copy)",
        )


class _CudaIpcPeerCopyOperation:
    """One rank of a source-driven, strongly completed intra-node copy."""

    def __init__(
        self,
        *,
        context: Mapping[str, Any],
        options: Mapping[str, Any],
        family: str,
        input_splits: list[int],
        output_splits: list[int],
        replicate_source: bool = False,
    ) -> None:
        self.context = dict(context)
        self.family = family
        self.torch = context["torch"]
        self.dist = context["dist"]
        self.group = context["group"]
        self.group_rank = int(context["group_rank"])
        self.group_size = int(context["group_size"])
        self.source = context["source"]
        self.output = context["output"]
        self.input_splits = [int(value) for value in input_splits]
        self.output_splits = [int(value) for value in output_splits]
        self.replicate_source = bool(replicate_source)
        if len(self.input_splits) != self.group_size:
            raise ValueError(f"{family} input split count is not group_size")
        if len(self.output_splits) != self.group_size:
            raise ValueError(f"{family} output split count is not group_size")
        if any(value < 0 for value in self.input_splits + self.output_splits):
            raise ValueError(f"{family} split sizes must be nonnegative")
        if self.replicate_source:
            if any(value != int(self.source.numel()) for value in self.input_splits):
                raise ValueError(
                    f"{family} replicated input sizes do not match source tensor"
                )
        elif sum(self.input_splits) != int(self.source.numel()):
            raise ValueError(f"{family} input splits do not cover source tensor")
        if sum(self.output_splits) != int(self.output.numel()):
            raise ValueError(f"{family} output splits do not cover output tensor")
        if self.source.dtype != self.output.dtype:
            raise ValueError(f"{family} source/output dtypes differ")
        if not self.source.is_contiguous() or not self.output.is_contiguous():
            raise ValueError(f"{family} peer-copy tensors must be contiguous")

        element_bytes = int(self.source.element_size())
        output_nbytes = int(self.output.numel()) * element_bytes
        self.runtime = _CudaRuntime(
            str(options.get("cuda_runtime_library", "libcudart.so.12")),
            str(options.get("cuda_driver_library", "libcuda.so.1")),
        )
        self.stream = self.torch.cuda.Stream(priority=0)
        self.source_done = self.torch.cuda.Event(interprocess=True)
        local_record = {
            "group_rank": self.group_rank,
            "input_splits": self.input_splits,
            "output_splits": self.output_splits,
            "dtype": str(self.source.dtype),
            "element_bytes": element_bytes,
            "output": self.runtime.export_tensor(
                int(self.output.data_ptr()), output_nbytes
            ),
            "source_done_event_hex": self.source_done.ipc_handle().hex(),
        }
        records: list[Any] = [None] * self.group_size
        self.dist.all_gather_object(records, local_record, group=self.group)
        for rank, record in enumerate(records):
            if not isinstance(record, Mapping) or int(record["group_rank"]) != rank:
                raise RuntimeError(f"{family} IPC rendezvous rank ordering changed")
            if str(record["dtype"]) != str(self.source.dtype):
                raise RuntimeError(f"{family} IPC rendezvous dtype mismatch")
            if int(record["element_bytes"]) != element_bytes:
                raise RuntimeError(f"{family} IPC rendezvous element-size mismatch")
            if len(record["input_splits"]) != self.group_size:
                raise RuntimeError(f"{family} remote input split count mismatch")
            if len(record["output_splits"]) != self.group_size:
                raise RuntimeError(f"{family} remote output split count mismatch")
        for source in range(self.group_size):
            for destination in range(self.group_size):
                sent = int(records[source]["input_splits"][destination])
                received = int(records[destination]["output_splits"][source])
                if sent != received:
                    raise RuntimeError(
                        f"{family} split transpose mismatch at "
                        f"source={source}, destination={destination}: {sent} != {received}"
                    )

        self._mapping_bases: list[int] = []
        self._destination_pointers: list[int] = []
        try:
            for destination, record in enumerate(records):
                if destination == self.group_rank:
                    self._destination_pointers.append(int(self.output.data_ptr()))
                else:
                    base, pointer = self.runtime.open_tensor(record["output"])
                    self._mapping_bases.append(base)
                    self._destination_pointers.append(pointer)
        except BaseException:
            for base in reversed(self._mapping_bases):
                self.runtime.close_mapping(base)
            raise

        self._completion_events = []
        for source, record in enumerate(records):
            if source == self.group_rank:
                event = self.source_done
            else:
                event = self.torch.cuda.Event.from_ipc_handle(
                    0, bytes.fromhex(str(record["source_done_event_hex"]))
                )
            self._completion_events.append(event)
        self._transfers: list[tuple[int, int, int]] = []
        source_offset = 0
        for destination, count in enumerate(self.input_splits):
            destination_splits = [
                int(value) for value in records[destination]["output_splits"]
            ]
            destination_offset = sum(destination_splits[: self.group_rank])
            if count:
                self._transfers.append((
                    int(self.source.data_ptr()) + source_offset * element_bytes,
                    self._destination_pointers[destination]
                    + destination_offset * element_bytes,
                    count * element_bytes,
                ))
            if not self.replicate_source:
                source_offset += count
        self._generation = 0
        self._outstanding: int | None = None
        self._closed = False

    def launch(self) -> int:
        if self._closed:
            raise RuntimeError(f"{self.family} operation is closed")
        if self._outstanding is not None:
            raise RuntimeError(f"{self.family} launch called before prior wait")
        self._generation += 1
        generation = self._generation
        self.stream.wait_stream(self.torch.cuda.current_stream())
        for source, destination, nbytes in self._transfers:
            self.runtime.copy_async(
                destination=destination,
                source=source,
                nbytes=nbytes,
                stream=int(self.stream.cuda_stream),
            )
        self.source_done.record(self.stream)
        self._outstanding = generation
        return generation

    def wait(self, result: Any) -> None:
        generation = int(result)
        if self._outstanding != generation:
            raise RuntimeError(
                f"{self.family} wait token mismatch: "
                f"outstanding={self._outstanding}, result={generation}"
            )
        # Waiting for every source's post-copy event is deliberately stronger
        # than local receive visibility: it also establishes source reuse for
        # all ranks, matching the accepted isolated peer-copy completion rule.
        for event in self._completion_events:
            event.synchronize()
        self._outstanding = None

    def close(self) -> None:
        if self._closed:
            return
        if self._outstanding is not None:
            raise RuntimeError(f"cannot close outstanding {self.family} operation")
        for base in reversed(self._mapping_bases):
            self.runtime.close_mapping(base)
        self._mapping_bases.clear()
        self._closed = True

    def plugin(self) -> dict[str, Any]:
        return {
            "launch": self.launch,
            "wait": self.wait,
            "close": self.close,
            "owned": [self],
            "timing_domain": "host",
            "metadata": {
                "implementation": "host_cuda_ipc_peer_copy",
                "isolated_implementation_api": "cudaMemcpyPeerAsync",
                "suite_adapter_api": "cudaMemcpyAsync(cudaMemcpyDefault)",
                "peer_mapping": "cudaIpcOpenMemHandle",
                "copy_plan": "source_driven",
                "completion": (
                    "host synchronize every source interprocess CUDA event; "
                    "destination visible and every source reusable"
                ),
                "launches_sm_kernel": False,
                "timing_domain": "host",
            },
        }


class _NcclZeroCtaOperation:
    """One raw NCCL 2.28.3 collective on the suite-managed CUDA stream."""

    def __init__(
        self,
        *,
        context: Mapping[str, Any],
        options: Mapping[str, Any],
        family: str,
    ) -> None:
        overlay, expected, compat_path = _zero_cta_settings(family, options)
        if _PREPARED_ZERO_CTA != (overlay, expected, compat_path):
            raise RuntimeError(
                f"{family} Zero-CTA process setup did not run before Torch import"
            )
        import nccl.core as nccl

        versions = nccl.get_version()
        if str(versions.nccl_version) != expected:
            raise RuntimeError(
                f"{family} expected NCCL {expected}, got {versions.nccl_version}"
            )
        self.nccl = nccl
        self.torch = context["torch"]
        self.family = family
        self.source = context["source"]
        self.output = context["output"]
        self.group_rank = int(context["group_rank"])
        self.group_size = int(context["group_size"])
        self.closed = False
        if not self.source.is_cuda or not self.output.is_cuda:
            raise ValueError(f"{family} Zero-CTA requires CUDA tensors")
        if not self.source.is_contiguous() or not self.output.is_contiguous():
            raise ValueError(f"{family} Zero-CTA requires contiguous tensors")
        if self.source.dtype != self.output.dtype:
            raise ValueError(f"{family} source/output dtype mismatch")
        if family == "all_gather":
            expected_output = self.source.numel() * self.group_size
            if self.output.numel() != expected_output:
                raise ValueError(
                    f"all_gather output has {self.output.numel()} elements, "
                    f"expected {expected_output}"
                )
        elif (
            self.source.numel() != self.output.numel()
            or self.source.numel() % self.group_size
        ):
            raise ValueError(
                "balanced_a2a requires equal input/output sizes divisible by 8"
            )

        uid_bytes = (
            bytes(nccl.get_unique_id().as_bytes)
            if self.group_rank == 0
            else None
        )
        gathered: list[Any] = [None] * self.group_size
        context["dist"].all_gather_object(
            gathered, uid_bytes, group=context["group"]
        )
        if not isinstance(gathered[0], bytes) or any(
            value is not None for value in gathered[1:]
        ):
            raise RuntimeError(
                f"{family} NCCL unique-ID rendezvous was inconsistent: "
                f"{[type(value).__name__ for value in gathered]}"
            )
        self.config = nccl.NCCLConfig(
            cta_policy=nccl.CTAPolicy.Zero,
            comm_name=(
                "overlap_ag_zero_cta"
                if family == "all_gather"
                else "overlap_a2a_zero_cta"
            ),
        )
        self.comm = nccl.Communicator.init(
            self.group_size,
            self.group_rank,
            nccl.UniqueId.from_bytes(gathered[0]),
            self.config,
        )
        self.source_window = self.comm.register_window(
            self.source, nccl.WindowFlag.CollSymmetric
        )
        self.output_window = self.comm.register_window(
            self.output, nccl.WindowFlag.CollSymmetric
        )
        if self.source_window is None or self.output_window is None:
            self.comm.abort()
            raise RuntimeError(
                f"{family}: NCCL CollSymmetric window registration returned NULL"
            )

    def launch(self) -> None:
        if self.closed:
            raise RuntimeError(f"{self.family} Zero-CTA communicator is closed")
        stream = int(self.torch.cuda.current_stream().cuda_stream)
        if self.family == "all_gather":
            self.comm.allgather(self.source, self.output, stream=stream)
        else:
            self.comm.alltoall(self.source, self.output, stream=stream)

    def close(self) -> None:
        if self.closed:
            return
        self.comm.destroy()
        self.closed = True

    def plugin(self) -> dict[str, Any]:
        peer_numel = (
            self.source.numel()
            if self.family == "all_gather"
            else self.source.numel() // self.group_size
        )
        return {
            "launch": self.launch,
            "close": self.close,
            "owned": [self],
            "timing_domain": "cuda_event",
            "metadata": {
                "implementation": f"nccl_2_28_3_zero_cta_{self.family}",
                "communication_api": (
                    "ncclAllGather"
                    if self.family == "all_gather"
                    else "ncclAllToAll"
                ),
                "nccl_runtime_version": "2.28.3",
                "nccl_cta_policy": "Zero",
                "nccl_cta_policy_api": (
                    "NCCLConfig(cta_policy=CTAPolicy.Zero)"
                ),
                "symmetric_window_flag": "CollSymmetric",
                "allocator": "ncclMemAlloc via nccl.core.interop.torch.empty",
                "peer_numel": int(peer_numel),
                "launches_sm_kernel_expected": False,
                "timing_domain": "cuda_event",
            },
        }


def _allocate_zero_cta_buffers(
    *, context: Mapping[str, Any], options: Mapping[str, Any]
) -> dict[str, Any]:
    family = str(context.get("family"))
    _require_context(
        context,
        family=family,
        group_size=8,
        names=(
            "torch", "group_rank", "group_size", "source_numel",
            "output_numel", "dtype",
        ),
    )
    overlay, expected, compat_path = _zero_cta_settings(family, options)
    if _PREPARED_ZERO_CTA != (overlay, expected, compat_path):
        raise RuntimeError(
            f"{family} Zero-CTA allocator was called before process setup"
        )
    from nccl.core.interop.torch import empty as nccl_empty

    source_numel = int(context["source_numel"])
    output_numel = int(context["output_numel"])
    if source_numel <= 0 or output_numel <= 0:
        raise ValueError(f"{family} Zero-CTA buffers must be nonempty")
    torch = context["torch"]
    device = torch.cuda.current_device()
    source = nccl_empty(
        source_numel, dtype=context["dtype"], device=device
    )
    output = nccl_empty(
        output_numel, dtype=context["dtype"], device=device
    )
    return {
        "source": source,
        "output": output,
        "owned": [source, output],
        "metadata": {
            "buffer_allocator": "ncclMemAlloc",
            "source_numel": source_numel,
            "output_numel": output_numel,
        },
    }


def create_all_gather_nccl_228_zero_cta(
    *, context: Mapping[str, Any], options: Mapping[str, Any]
) -> dict[str, Any]:
    """Create NCCL 2.28.3's registered-window Zero-CTA all-gather."""

    _require_context(
        context,
        family="all_gather",
        group_size=8,
        names=(
            "torch", "dist", "group", "group_rank", "group_size",
            "source", "output", "input_numel",
        ),
    )
    operation = _NcclZeroCtaOperation(
        context=context, options=options, family="all_gather"
    )
    return operation.plugin()


def create_balanced_a2a_nccl_228_zero_cta(
    *, context: Mapping[str, Any], options: Mapping[str, Any]
) -> dict[str, Any]:
    """Create NCCL 2.28.3's registered-window Zero-CTA balanced A2A."""

    _require_context(
        context,
        family="balanced_a2a",
        group_size=8,
        names=(
            "torch", "dist", "group", "group_rank", "group_size",
            "source", "output", "input_split_sizes", "output_split_sizes",
        ),
    )
    input_splits = [int(value) for value in context["input_split_sizes"]]
    output_splits = [int(value) for value in context["output_split_sizes"]]
    if len(set(input_splits + output_splits)) != 1:
        raise ValueError(
            "balanced_a2a Zero-CTA requires one equal count for every peer"
        )
    operation = _NcclZeroCtaOperation(
        context=context, options=options, family="balanced_a2a"
    )
    return operation.plugin()


def create_all_gather_torch_distributed_nccl_default_cta(
    *, context: Mapping[str, Any], options: Mapping[str, Any]
) -> dict[str, Any]:
    """Re-expose the suite's normal all-gather through the plugin contract."""
    _options("all_gather", options, frozenset())
    _require_context(
        context,
        family="all_gather",
        group_size=8,
        names=("dist", "group", "source", "output", "input_numel"),
    )
    dist = context["dist"]
    source = context["source"]
    output = context["output"]
    group = context["group"]

    def launch() -> Any:
        return dist.all_gather_into_tensor(
            output, source, group=group, async_op=True
        )

    return {
        "launch": launch,
        "owned": [],
        "timing_domain": "cuda_event",
        "metadata": {
            "implementation": "torch_distributed_nccl_default_cta_all_gather",
            "communication_api": "torch.distributed.all_gather_into_tensor",
            "nccl_cta_policy": "Default",
            "buffer_allocator": "torch.empty",
            "timing_domain": "cuda_event",
        },
    }


def _allocate_multimem_all_gather_buffers(
    *, context: Mapping[str, Any], options: Mapping[str, Any]
) -> dict[str, Any]:
    """Allocate the multicast destination required by multimem all-gather."""

    parsed = _options("all_gather", options, _MULTIMEM_OPTIONS)
    if parsed.get("use_physical_cuda_ordinals") is not True:
        raise ValueError(
            "multimem all-gather requires use_physical_cuda_ordinals=true"
        )
    _require_context(
        context,
        family="all_gather",
        group_size=8,
        names=(
            "torch", "group_rank", "group_size", "source_numel",
            "output_numel", "dtype",
        ),
    )
    torch = context["torch"]
    source_numel = int(context["source_numel"])
    output_numel = int(context["output_numel"])
    if source_numel <= 0 or output_numel != source_numel * 8:
        raise ValueError(
            "multimem all-gather requires a nonempty rank-major 8x output"
        )
    import torch.distributed._symmetric_memory as symm_mem

    device = torch.cuda.current_device()
    # The backend is process-global and cannot be changed after first use.
    # This Torch build selects CUDA by default in the benchmark workers; it
    # rejects a redundant set_backend("CUDA"), so assert instead of resetting.
    active_backend = symm_mem._SymmetricMemory.get_backend(
        torch.device("cuda", device)
    )
    if active_backend != "CUDA":
        raise RuntimeError(
            "multimem all-gather requires the CUDA symmetric-memory backend; "
            f"active backend is {active_backend!r}"
        )
    source = torch.empty(source_numel, dtype=context["dtype"], device=device)
    output = symm_mem.empty(
        output_numel, dtype=context["dtype"], device=device
    )
    return {
        "source": source,
        "output": output,
        "owned": [source, output],
        "metadata": {
            "buffer_allocator": "torch symmetric-memory CUDA backend",
            "symmetric_memory_backend": active_backend,
            "source_numel": source_numel,
            "output_numel": output_numel,
            "multicast_destination_requested": True,
        },
    }


class _MultimemAllGatherOperation:
    """PyTorch's multicast-memory 8-rank all-gather on the current stream."""

    def __init__(
        self, *, context: Mapping[str, Any], options: Mapping[str, Any]
    ) -> None:
        parsed = _options("all_gather", options, _MULTIMEM_OPTIONS)
        if parsed.get("use_physical_cuda_ordinals") is not True:
            raise ValueError(
                "multimem all-gather requires use_physical_cuda_ordinals=true"
            )
        _require_context(
            context,
            family="all_gather",
            group_size=8,
            names=(
                "torch", "dist", "group", "group_rank", "group_size",
                "source", "output", "input_numel",
            ),
        )
        self.torch = context["torch"]
        self.source = context["source"]
        self.output = context["output"]
        self.group = context["group"]
        input_numel = int(context["input_numel"])
        if not self.source.is_cuda or not self.output.is_cuda:
            raise ValueError("multimem all-gather requires CUDA tensors")
        if not self.source.is_contiguous() or not self.output.is_contiguous():
            raise ValueError("multimem all-gather requires contiguous tensors")
        if self.source.dtype != self.output.dtype:
            raise ValueError("multimem all-gather source/output dtype mismatch")
        if (
            self.source.numel() != input_numel
            or self.output.numel() != input_numel * 8
        ):
            raise ValueError("multimem all-gather tensor geometry changed")

        import torch.distributed._symmetric_memory as symm_mem
        from torch._C._autograd import DeviceType

        device_index = int(self.source.device.index)
        if not symm_mem._SymmetricMemory.has_multicast_support(
            DeviceType.CUDA, device_index
        ):
            raise RuntimeError(
                f"CUDA device {device_index} lacks symmetric-memory multicast support"
            )
        self.handle = symm_mem.rendezvous(self.output, group=self.group)
        multicast_ptr = int(self.handle.multicast_ptr)
        if multicast_ptr == 0:
            raise RuntimeError("symmetric-memory rendezvous returned no multicast pointer")
        self.group_name = str(self.group.group_name)
        self.multicast_ptr = multicast_ptr

    def launch(self) -> Any:
        return self.torch.ops.symm_mem.multimem_all_gather_out(
            self.source, self.group_name, self.output
        )

    def plugin(self) -> dict[str, Any]:
        return {
            "launch": self.launch,
            "owned": [self.handle],
            "timing_domain": "cuda_event",
            "metadata": {
                "implementation": "torch_symm_mem_multimem_all_gather_out",
                "communication_api": "torch.ops.symm_mem.multimem_all_gather_out",
                "symmetric_memory_backend": "CUDA",
                "multicast_support": True,
                "multicast_pointer_nonzero": True,
                "rank_order": "rank-major contiguous output",
                "timing_domain": "cuda_event",
            },
        }


def create_all_gather_torch_symm_mem_multimem(
    *, context: Mapping[str, Any], options: Mapping[str, Any]
) -> dict[str, Any]:
    """Create PyTorch's existing multicast-memory all-gather operator."""

    return _MultimemAllGatherOperation(
        context=context, options=options
    ).plugin()


def create_balanced_a2a_torch_distributed_nccl_default_cta(
    *, context: Mapping[str, Any], options: Mapping[str, Any]
) -> dict[str, Any]:
    """Re-expose the suite's normal balanced A2A through the plugin contract."""
    _options("balanced_a2a", options, frozenset())
    _require_context(
        context,
        family="balanced_a2a",
        group_size=8,
        names=(
            "dist", "group", "source", "output",
            "input_split_sizes", "output_split_sizes",
        ),
    )
    dist = context["dist"]
    source = context["source"]
    output = context["output"]
    group = context["group"]
    input_splits = [int(value) for value in context["input_split_sizes"]]
    output_splits = [int(value) for value in context["output_split_sizes"]]
    if len(set(input_splits + output_splits)) != 1:
        raise ValueError("normal balanced A2A requires equal splits")

    def launch() -> Any:
        return dist.all_to_all_single(
            output,
            source,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group=group,
            async_op=True,
        )

    return {
        "launch": launch,
        "owned": [],
        "timing_domain": "cuda_event",
        "metadata": {
            "implementation": "torch_distributed_nccl_default_cta_balanced_a2a",
            "communication_api": "torch.distributed.all_to_all_single",
            "nccl_cta_policy": "Default",
            "buffer_allocator": "torch.empty",
            "timing_domain": "cuda_event",
        },
    }


# The generic suite contract discovers this optional allocation hook on the
# selected factory before constructing validation closures.
create_all_gather_nccl_228_zero_cta.allocate = _allocate_zero_cta_buffers
create_balanced_a2a_nccl_228_zero_cta.allocate = _allocate_zero_cta_buffers
create_all_gather_torch_symm_mem_multimem.allocate = (
    _allocate_multimem_all_gather_buffers
)


def create_all_gather_host_peer_copy(
    *, context: Mapping[str, Any], options: Mapping[str, Any]
) -> dict[str, Any]:
    """Create the selected 8-way rank-major host peer-copy all-gather."""

    _require_context(
        context,
        family="all_gather",
        group_size=8,
        names=(
            "torch", "dist", "group", "group_rank", "group_size",
            "source", "output", "input_numel",
        ),
    )
    parsed = _options("all_gather", options, _COMMON_COPY_OPTIONS)
    input_numel = int(context["input_numel"])
    if input_numel <= 0:
        raise ValueError("all_gather input_numel must be positive")
    operation = _CudaIpcPeerCopyOperation(
        context=context,
        options=parsed,
        family="all_gather",
        input_splits=[input_numel] * 8,
        output_splits=[input_numel] * 8,
        replicate_source=True,
    )
    return operation.plugin()


def create_ragged_a2a_host_peer_copy(
    *, context: Mapping[str, Any], options: Mapping[str, Any]
) -> dict[str, Any]:
    """Create the selected source-ordered 8-way ragged host peer copy."""

    _require_context(
        context,
        family="ragged_a2a",
        group_size=8,
        names=(
            "torch", "dist", "group", "group_rank", "group_size",
            "source", "output", "input_split_sizes", "output_split_sizes",
        ),
    )
    parsed = _options("ragged_a2a", options, _COMMON_COPY_OPTIONS)
    operation = _CudaIpcPeerCopyOperation(
        context=context,
        options=parsed,
        family="ragged_a2a",
        input_splits=[int(value) for value in context["input_split_sizes"]],
        output_splits=[int(value) for value in context["output_split_sizes"]],
    )
    return operation.plugin()


__all__ = [
    "ALL_GATHER_NORMAL_SELECTOR",
    "ALL_GATHER_SELECTOR",
    "MULTIMEM_ALL_GATHER_SELECTOR",
    "BALANCED_A2A_NORMAL_SELECTOR",
    "BALANCED_A2A_SELECTOR",
    "RAGGED_A2A_SELECTOR",
    "create_all_gather_nccl_228_zero_cta",
    "create_all_gather_torch_distributed_nccl_default_cta",
    "create_all_gather_torch_symm_mem_multimem",
    "create_balanced_a2a_nccl_228_zero_cta",
    "create_balanced_a2a_torch_distributed_nccl_default_cta",
    "create_all_gather_host_peer_copy",
    "create_ragged_a2a_host_peer_copy",
]
