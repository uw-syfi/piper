from __future__ import annotations

import os
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import torch


class Device(ABC):
    """Abstract interface over CUDA vs CPU runtime primitives."""

    @abstractmethod
    def create_stream_context(self) -> AbstractContextManager:
        """Return a reusable stream context manager."""

    @abstractmethod
    def create_event(self) -> Any:
        """Create a synchronization event (or None on CPU)."""

    @abstractmethod
    def record_event(self, stream_ctx: AbstractContextManager) -> Any:
        """Create and record an event on *stream_ctx*."""

    @abstractmethod
    def wait_event(self, stream_ctx: AbstractContextManager, evt: Any) -> None:
        """Make *stream_ctx* wait for *evt*."""

    @abstractmethod
    def synchronize(self) -> None:
        """Block until all device work completes."""

    @abstractmethod
    def set_device(self, global_rank: int) -> str:
        """Set the active device for this rank and return the device string."""

    @property
    @abstractmethod
    def dist_backend(self) -> str:
        """Return the torch.distributed backend name."""

    @property
    @abstractmethod
    def accelerator_resource(self) -> str | None:
        """Name of the Ray accelerator resource (e.g. ``"GPU"``), or None if none is needed."""

    @abstractmethod
    def profiler_activities(self) -> list:
        """Return the list of profiler activities to trace."""

    @abstractmethod
    def get_and_reset_peak_memory(self) -> int | None:
        """Return peak memory allocated in bytes (and reset), or None on CPU."""

    @abstractmethod
    def reset_peak_memory(self) -> None:
        """Reset peak memory stats."""

    @abstractmethod
    def nvtx_push(self, label: str) -> None:
        """Push an NVTX range (no-op on CPU)."""

    @abstractmethod
    def nvtx_pop(self) -> None:
        """Pop an NVTX range (no-op on CPU)."""



class CpuDevice(Device):

    def create_stream_context(self) -> AbstractContextManager:
        return nullcontext()

    def create_event(self) -> Any:
        return None

    def record_event(self, stream_ctx: AbstractContextManager) -> Any:
        return None

    def wait_event(self, stream_ctx: AbstractContextManager, evt: Any) -> None:
        pass

    def synchronize(self) -> None:
        pass

    def set_device(self, global_rank: int) -> str:
        return "cpu"

    @property
    def dist_backend(self) -> str:
        return "gloo"

    @property
    def accelerator_resource(self) -> str | None:
        return None

    def profiler_activities(self) -> list:
        return [torch.profiler.ProfilerActivity.CPU]

    def get_and_reset_peak_memory(self) -> int | None:
        # TODO(swang): Is there a way to get peak torch CPU memory?
        return None

    def reset_peak_memory(self) -> None:
        pass

    def nvtx_push(self, label: str) -> None:
        pass

    def nvtx_pop(self) -> None:
        pass


class CudaDevice(Device):

    def create_stream_context(self) -> AbstractContextManager:
        stream = torch.cuda.Stream()
        ctx = torch.cuda.stream(stream)
        # Force cuBLAS context initialization so the first backward pass
        # does not hit lazy CUDA warnings.
        with ctx:
            w = torch.zeros(4, 4, device=stream.device)
            torch.mm(w, w)
        return ctx

    def create_event(self) -> Any:
        return torch.cuda.Event()

    def record_event(self, stream_ctx: AbstractContextManager) -> Any:
        evt = torch.cuda.Event()
        evt.record(stream_ctx.stream)
        return evt

    def wait_event(self, stream_ctx: AbstractContextManager, evt: Any) -> None:
        if evt is not None:
            stream_ctx.stream.wait_event(evt)

    def synchronize(self) -> None:
        torch.cuda.synchronize()

    def set_device(self, global_rank: int) -> str:
        device = f"cuda:{global_rank % torch.cuda.device_count()}"
        torch.cuda.set_device(device)
        return device

    @property
    def dist_backend(self) -> str:
        return "nccl"

    @property
    def accelerator_resource(self) -> str | None:
        return "GPU"

    def profiler_activities(self) -> list:
        return [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]

    def get_and_reset_peak_memory(self) -> int | None:
        max_alloc = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        return max_alloc

    def reset_peak_memory(self) -> None:
        torch.cuda.reset_peak_memory_stats()

    def nvtx_push(self, label: str) -> None:
        torch.cuda.nvtx.range_push(label)

    def nvtx_pop(self) -> None:
        torch.cuda.nvtx.range_pop()


def piper_device() -> str:
    """Return the Piper execution device (``"cpu"`` or ``"cuda"``)."""
    return os.environ.get("PIPER_DEVICE", "cuda")


_device_instance: Device | None = None


def get_device() -> Device:
    """Return the singleton ``Device`` for the current process."""
    global _device_instance
    if _device_instance is None:
        if piper_device().startswith("cuda"):
            _device_instance = CudaDevice()
        else:
            _device_instance = CpuDevice()
    return _device_instance
