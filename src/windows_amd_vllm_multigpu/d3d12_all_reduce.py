"""Two-rank GPU-driven all-reduce over D3D12 cross-adapter heaps."""

from __future__ import annotations

import os
import uuid

import torch
import torch.distributed as dist

from .d3d12_shared import D3D12SharedBuffer
from .hip_runtime import HipRuntime
from .mapped_peer_kernel import MappedPeerKernel


def d3d12_requested() -> bool:
    return os.environ.get("WAVMG_USE_D3D12", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class D3D12AllReduce:
    """Stream-ordered SUM for exactly two HIP ranks without CPU payload copies."""

    def __init__(
        self,
        group: dist.ProcessGroup | None,
        device: torch.device | str | int,
        max_size_bytes: int | None = None,
        min_size_bytes: int | None = None,
    ) -> None:
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized first")
        self.group = group
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        if self.world_size != 2:
            raise ValueError("D3D12AllReduce currently requires world size 2")
        self.device = (
            torch.device("cuda", device)
            if isinstance(device, int)
            else torch.device(device)
        )
        if self.device.type != "cuda":
            raise ValueError(f"expected a HIP/CUDA-like device, got {self.device}")
        configured_size = int(
            os.environ.get("WAVMG_D3D12_MAX_BYTES", str(64 * 1024 * 1024))
        )
        self.max_size_bytes = (
            configured_size if max_size_bytes is None else max_size_bytes
        )
        if self.max_size_bytes <= 0:
            raise ValueError("D3D12 maximum payload must be positive")
        configured_min_size = int(
            os.environ.get("WAVMG_D3D12_MIN_BYTES", str(32 * 1024))
        )
        self.min_size_bytes = (
            configured_min_size if min_size_bytes is None else min_size_bytes
        )
        if self.min_size_bytes < 0:
            raise ValueError("D3D12 minimum payload cannot be negative")
        if self.min_size_bytes > self.max_size_bytes:
            raise ValueError("D3D12 minimum payload cannot exceed its maximum payload")

        torch.cuda.set_device(self.device)
        descriptor: list[str | None] = [None]
        if self.rank == 0:
            descriptor[0] = f"Local\\wavmg-d3d12-{uuid.uuid4().hex}"
        global_source = dist.get_global_rank(group, 0) if group is not None else 0
        dist.broadcast_object_list(descriptor, src=global_source, group=group)
        if descriptor[0] is None:
            raise RuntimeError("rank 0 did not broadcast a D3D12 object name")
        self.base_name = descriptor[0]
        self._runtime = HipRuntime()
        self._kernel = MappedPeerKernel()
        self._own: D3D12SharedBuffer | None = None
        self._peer: D3D12SharedBuffer | None = None
        self._epoch = 0

        local_error: Exception | None = None
        try:
            self._own = D3D12SharedBuffer(
                f"{self.base_name}.rank{self.rank}",
                self.max_size_bytes,
                self.device.index if self.device.index is not None else self.rank,
                create=True,
            )
        except Exception as error:
            local_error = error
        self._agree_or_raise(local_error, "create D3D12 cross-adapter heap")

        local_error = None
        try:
            self._peer = D3D12SharedBuffer(
                f"{self.base_name}.rank{1 - self.rank}",
                self.max_size_bytes,
                self.device.index if self.device.index is not None else self.rank,
                create=False,
            )
        except Exception as error:
            local_error = error
        self._agree_or_raise(local_error, "open peer D3D12 cross-adapter heap")

    def _agree_or_raise(self, local_error: Exception | None, operation: str) -> None:
        available = torch.tensor(
            int(local_error is None), dtype=torch.int32, device="cpu"
        )
        dist.all_reduce(available, op=dist.ReduceOp.MIN, group=self.group)
        if available.item():
            return
        self._close_buffers()
        detail = f": {local_error}" if local_error is not None else " on peer rank"
        raise RuntimeError(f"failed to {operation}{detail}")

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._own is None or self._peer is None:
            raise RuntimeError("D3D12 all-reduce is closed")
        if tensor.device != self.device:
            raise ValueError(
                f"tensor is on {tensor.device}, communicator is {self.device}"
            )
        source = tensor if tensor.is_contiguous() else tensor.contiguous()
        if source.dtype not in (torch.float16, torch.float32, torch.bfloat16):
            raise TypeError(f"D3D12 all-reduce does not support {source.dtype}")
        size_bytes = source.numel() * source.element_size()
        if size_bytes == 0:
            return torch.empty_like(source)
        if size_bytes > self.max_size_bytes:
            raise ValueError(
                f"collective payload {size_bytes} exceeds "
                f"WAVMG_D3D12_MAX_BYTES={self.max_size_bytes}"
            )

        self._epoch += 1
        ready_value = (self._epoch * 2) - 1
        consumed_value = ready_value + 1
        # HIP's current device is thread-local. vLLM/Inductor may invoke this
        # custom collective from a thread whose current device is not the one
        # selected when the communicator was constructed. Raw ctypes HIP calls
        # do not perform PyTorch's usual per-tensor device guard, so establish
        # the owning device before obtaining the stream or touching pointers.
        torch.cuda.set_device(self.device)
        device_index = self.device.index
        if device_index is None:
            raise RuntimeError("D3D12 communicator device must have an index")
        self._runtime.set_device(device_index)
        stream = torch.cuda.current_stream(self.device)
        stream_pointer = int(stream.cuda_stream)
        output = torch.empty_like(source)
        self._kernel.copy_async(source, self._own.device_pointer, stream=stream)
        self._own.signal(ready_value, stream_pointer)
        self._peer.wait(ready_value, stream_pointer)
        self._kernel.add_async(source, self._peer.device_pointer, output, stream=stream)
        self._peer.signal(consumed_value, stream_pointer)
        self._own.wait(consumed_value, stream_pointer)
        return output

    def can_handle(self, tensor: torch.Tensor) -> bool:
        """Return whether this tensor is in the tuned D3D12 payload range."""
        size_bytes = tensor.numel() * tensor.element_size()
        return (
            tensor.dtype in (torch.float16, torch.float32, torch.bfloat16)
            and 0 < size_bytes
            and self.min_size_bytes <= size_bytes <= self.max_size_bytes
        )

    def _close_buffers(self) -> None:
        for name in ("_peer", "_own"):
            shared = getattr(self, name, None)
            if shared is not None:
                shared.close()
                setattr(self, name, None)

    def destroy(self) -> None:
        if self._own is None and self._peer is None:
            return
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        self._close_buffers()
