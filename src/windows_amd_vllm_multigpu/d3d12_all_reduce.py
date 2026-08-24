"""Two-rank GPU-driven all-reduce over D3D12 cross-adapter heaps."""

from __future__ import annotations

import os
import uuid

import torch
import torch.distributed as dist

from .d3d12_shared import D3D12SharedBuffer
from .hip_runtime import HIP_MEMCPY_DEFAULT, HipRuntime
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
            os.environ.get("WAVMG_D3D12_MAX_BYTES", 64 * 1024 * 1024)
        )
        self.max_size_bytes = max_size_bytes or configured_size
        if self.max_size_bytes <= 0:
            raise ValueError("D3D12 maximum payload must be positive")

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
            raise ValueError(f"tensor is on {tensor.device}, communicator is {self.device}")
        source = tensor if tensor.is_contiguous() else tensor.contiguous()
        if source.dtype not in (torch.float16, torch.float32, torch.bfloat16):
            raise TypeError(f"D3D12 all-reduce does not support {source.dtype}")
        size_bytes = source.numel() * source.element_size()
        if size_bytes > self.max_size_bytes:
            raise ValueError(
                f"collective payload {size_bytes} exceeds "
                f"WAVMG_D3D12_MAX_BYTES={self.max_size_bytes}"
            )

        self._epoch += 1
        ready_value = (self._epoch * 2) - 1
        consumed_value = ready_value + 1
        stream = torch.cuda.current_stream(self.device)
        stream_pointer = int(stream.cuda_stream)
        peer_input = torch.empty_like(source)
        output = torch.empty_like(source)
        self._runtime.memcpy_async(
            self._own.device_pointer,
            source.data_ptr(),
            size_bytes,
            HIP_MEMCPY_DEFAULT,
            stream_pointer,
        )
        self._own.signal(ready_value, stream_pointer)
        self._peer.wait(ready_value, stream_pointer)
        self._runtime.memcpy_async(
            peer_input.data_ptr(),
            self._peer.device_pointer,
            size_bytes,
            HIP_MEMCPY_DEFAULT,
            stream_pointer,
        )
        self._kernel.add_async(
            source, peer_input.data_ptr(), output, stream=stream
        )
        # The ctypes HIP calls are invisible to PyTorch's caching allocator.
        # Prevent reuse of the staging allocation until this stream completes.
        peer_input.record_stream(stream)
        self._peer.signal(consumed_value, stream_pointer)
        self._own.wait(consumed_value, stream_pointer)
        return output

    def can_handle(self, tensor: torch.Tensor) -> bool:
        """Return whether this tensor fits the validated D3D12 fast path."""
        return (
            tensor.dtype in (torch.float16, torch.float32, torch.bfloat16)
            and tensor.numel() * tensor.element_size() <= self.max_size_bytes
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
