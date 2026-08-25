"""Single-process tensor transfer over a D3D12 cross-adapter heap."""

from __future__ import annotations

import os
import threading
import uuid

import torch

from .d3d12_shared import D3D12SharedBuffer
from .hip_runtime import HIP_MEMCPY_DEFAULT, HipRuntime


def _device(value: torch.device | str | int) -> torch.device:
    device = torch.device("cuda", value) if isinstance(value, int) else torch.device(value)
    if device.type != "cuda" or device.index is None:
        raise ValueError(f"expected an indexed HIP/CUDA-like device, got {device}")
    return device


class D3D12TensorBridge:
    """Copy tensors between two GPUs without a CPU-thread payload copy."""

    def __init__(
        self,
        source_device: torch.device | str | int,
        destination_device: torch.device | str | int,
        max_size_bytes: int | None = None,
    ) -> None:
        self.source_device = _device(source_device)
        self.destination_device = _device(destination_device)
        if self.source_device == self.destination_device:
            raise ValueError("source and destination devices must differ")
        self.max_size_bytes = max_size_bytes or int(
            os.environ.get("WAVMG_D3D12_MAX_BYTES", 64 * 1024 * 1024)
        )
        if self.max_size_bytes <= 0:
            raise ValueError("D3D12 maximum payload must be positive")

        self._runtime = HipRuntime()
        self._lock = threading.Lock()
        self._epoch = 0
        self._creator: D3D12SharedBuffer | None = None
        self._opener: D3D12SharedBuffer | None = None
        previous_device = torch.cuda.current_device()
        name = f"Local\\wavmg-transfer-{uuid.uuid4().hex}"
        try:
            self._creator = D3D12SharedBuffer(
                name,
                self.max_size_bytes,
                self.source_device.index,
                create=True,
            )
            self._opener = D3D12SharedBuffer(
                name,
                self.max_size_bytes,
                self.destination_device.index,
                create=False,
            )
        except Exception:
            self._close_buffers()
            raise
        finally:
            torch.cuda.set_device(previous_device)

    def can_handle(self, tensor: torch.Tensor) -> bool:
        return (
            tensor.device == self.source_device
            and tensor.numel() * tensor.element_size() <= self.max_size_bytes
        )

    def transfer(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._creator is None or self._opener is None:
            raise RuntimeError("D3D12 tensor bridge is closed")
        if tensor.device != self.source_device:
            raise ValueError(
                f"tensor is on {tensor.device}, bridge source is {self.source_device}"
            )
        source = tensor if tensor.is_contiguous() else tensor.contiguous()
        size_bytes = source.numel() * source.element_size()
        if size_bytes > self.max_size_bytes:
            raise ValueError(
                f"transfer payload {size_bytes} exceeds "
                f"WAVMG_D3D12_MAX_BYTES={self.max_size_bytes}"
            )
        destination = torch.empty_like(source, device=self.destination_device)
        if size_bytes == 0:
            return destination

        with self._lock:
            self._epoch += 1
            ready_value = (self._epoch * 2) - 1
            consumed_value = ready_value + 1
            source_stream = torch.cuda.current_stream(self.source_device)
            destination_stream = torch.cuda.default_stream(self.destination_device)
            previous_device = torch.cuda.current_device()
            try:
                torch.cuda.set_device(self.source_device)
                self._runtime.memcpy_async(
                    self._creator.device_pointer,
                    source.data_ptr(),
                    size_bytes,
                    HIP_MEMCPY_DEFAULT,
                    int(source_stream.cuda_stream),
                )
                self._creator.signal(ready_value, int(source_stream.cuda_stream))
                self._opener.wait(ready_value, int(destination_stream.cuda_stream))
                self._runtime.memcpy_async(
                    destination.data_ptr(),
                    self._opener.device_pointer,
                    size_bytes,
                    HIP_MEMCPY_DEFAULT,
                    int(destination_stream.cuda_stream),
                )
                self._opener.signal(
                    consumed_value, int(destination_stream.cuda_stream)
                )
                self._creator.wait(consumed_value, int(source_stream.cuda_stream))
            finally:
                torch.cuda.set_device(previous_device)
            source.record_stream(source_stream)
            destination.record_stream(destination_stream)
        return destination

    def _close_buffers(self) -> None:
        for name in ("_opener", "_creator"):
            shared = getattr(self, name, None)
            if shared is not None:
                shared.close()
                setattr(self, name, None)

    def close(self) -> None:
        if self._creator is None and self._opener is None:
            return
        with self._lock:
            previous_device = torch.cuda.current_device()
            try:
                torch.cuda.synchronize(self.source_device)
                torch.cuda.synchronize(self.destination_device)
                self._close_buffers()
            finally:
                torch.cuda.set_device(previous_device)

    def __enter__(self) -> D3D12TensorBridge:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
