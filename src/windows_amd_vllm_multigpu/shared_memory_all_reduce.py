"""Fast two-rank all-reduce using HIP-registered Windows shared memory."""

from __future__ import annotations

import ctypes
import os
from multiprocessing import shared_memory
import uuid

import torch
import torch.distributed as dist

from .hip_runtime import (
    HIP_HOST_REGISTER_MAPPED,
    HIP_MEMCPY_DEVICE_TO_HOST,
    HIP_MEMCPY_HOST_TO_DEVICE,
    HipRuntime,
)
from .mapped_peer_kernel import MappedPeerKernel
from .win32_barrier import Win32PairBarrier


class SharedMemoryAllReduce:
    """Stream-ordered SUM all-reduce optimized for two Windows GPU ranks."""

    _CONTROL_BYTES = 4096
    _CACHE_LINE_BYTES = 64

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
            raise ValueError("SharedMemoryAllReduce currently requires world size 2")
        if isinstance(device, int):
            device = torch.device("cuda", device)
        else:
            device = torch.device(device)
        if device.type != "cuda":
            raise ValueError(f"expected a HIP/CUDA-like device, got {device}")
        self.device = device

        configured_size = int(
            os.environ.get("WAVMG_SHM_MAX_BYTES", 64 * 1024 * 1024)
        )
        self.max_size_bytes = max_size_bytes or configured_size
        if self.max_size_bytes <= 0 or self.max_size_bytes % 4096:
            raise ValueError("max shared-memory size must be a positive 4 KiB multiple")

        descriptor: list[dict[str, object] | None] = [None]
        self._owns_mapping = self.rank == 0
        if self._owns_mapping:
            created = shared_memory.SharedMemory(
                create=True,
                size=(self.max_size_bytes * self.world_size) + self._CONTROL_BYTES,
            )
            descriptor[0] = {
                "mapping_name": created.name,
                "event_prefix": f"WAVMG_{uuid.uuid4().hex}",
                "max_size_bytes": self.max_size_bytes,
            }
            self._mapping = created

        global_source = dist.get_global_rank(group, 0) if group is not None else 0
        dist.broadcast_object_list(descriptor, src=global_source, group=group)
        value = descriptor[0]
        if value is None:
            raise RuntimeError("rank 0 did not broadcast a shared-memory descriptor")
        if int(value["max_size_bytes"]) != self.max_size_bytes:
            raise RuntimeError("shared-memory size differs between ranks")
        if not self._owns_mapping:
            self._mapping = shared_memory.SharedMemory(name=str(value["mapping_name"]))

        self._mapping_view = self._mapping.buf
        self._base_pointer = ctypes.addressof(
            ctypes.c_char.from_buffer(self._mapping_view)
        )
        self._runtime = HipRuntime()
        self._runtime.host_register(
            self._base_pointer,
            len(self._mapping_view),
            flags=HIP_HOST_REGISTER_MAPPED,
        )
        self._registered = True
        self._mapped_device_pointer = self._runtime.host_get_device_pointer(
            self._base_pointer
        )
        self._barrier = Win32PairBarrier(str(value["event_prefix"]), self.rank)
        self._slots = [
            torch.frombuffer(
                self._mapping_view,
                dtype=torch.uint8,
                count=self.max_size_bytes,
                offset=rank * self.max_size_bytes,
            )
            for rank in range(self.world_size)
        ]
        self._local_output = torch.empty(
            (self.max_size_bytes,), dtype=torch.uint8, pin_memory=True
        )
        stream_path_available = False
        try:
            self._kernel = MappedPeerKernel()
            device_index = self.device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            stream_path_available = self._runtime.can_use_stream_wait_value(
                device_index
            )
        except (AttributeError, FileNotFoundError, OSError, RuntimeError):
            self._kernel = None
        capability = torch.tensor(
            int(stream_path_available), dtype=torch.int32, device="cpu"
        )
        dist.all_reduce(capability, op=dist.ReduceOp.MIN, group=self.group)
        self._use_stream_path = bool(capability.item())
        self._epoch = 0
        self._barrier.wait()

    def _control_offset(self, kind: str, rank: int) -> int:
        control_base = self.max_size_bytes * self.world_size
        kind_index = 0 if kind == "ready" else self.world_size
        return control_base + (kind_index + rank) * self._CACHE_LINE_BYTES

    def _stream_all_reduce(
        self, source: torch.Tensor, size_bytes: int
    ) -> torch.Tensor:
        if self._kernel is None:
            raise RuntimeError("mapped-peer kernel was not initialized")
        if self._epoch >= 0xFFFFFF00:
            raise RuntimeError("stream epoch is near uint32 wrap; recreate communicator")
        self._epoch += 1
        epoch = self._epoch
        peer = 1 - self.rank
        stream = torch.cuda.current_stream(self.device)
        output = torch.empty_like(source)
        self._kernel.all_reduce_async(
            source,
            self._base_pointer + self.rank * self.max_size_bytes,
            self._mapped_device_pointer + peer * self.max_size_bytes,
            output,
            self._mapped_device_pointer + self._control_offset("ready", self.rank),
            self._mapped_device_pointer + self._control_offset("ready", peer),
            self._mapped_device_pointer
            + self._control_offset("consumed", self.rank),
            self._mapped_device_pointer + self._control_offset("consumed", peer),
            epoch,
            stream,
        )
        return output

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device != self.device:
            raise ValueError(f"tensor is on {tensor.device}, communicator is {self.device}")
        source = tensor if tensor.is_contiguous() else tensor.contiguous()
        size_bytes = source.numel() * source.element_size()
        if size_bytes > self.max_size_bytes:
            raise ValueError(
                f"collective payload {size_bytes} exceeds {self.max_size_bytes} bytes"
            )
        if self._use_stream_path and source.dtype in (
            torch.float32,
            torch.float16,
            torch.bfloat16,
        ):
            return self._stream_all_reduce(source, size_bytes)

        typed_slots = [
            slot[:size_bytes].view(source.dtype).view(source.shape)
            for slot in self._slots
        ]
        local_output = (
            self._local_output[:size_bytes].view(source.dtype).view(source.shape)
        )
        output = torch.empty_like(source)

        self._runtime.memcpy(
            typed_slots[self.rank].data_ptr(),
            source.data_ptr(),
            size_bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
        )
        self._barrier.wait()
        torch.add(typed_slots[0], typed_slots[1], out=local_output)
        self._runtime.memcpy(
            output.data_ptr(),
            local_output.data_ptr(),
            size_bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
        )
        self._barrier.wait()
        return output

    def destroy(self) -> None:
        if getattr(self, "_mapping", None) is None:
            return
        torch.cuda.synchronize(self.device)
        try:
            self._barrier.wait()
        finally:
            self._barrier.close()
            if self._registered:
                self._runtime.host_unregister(self._base_pointer)
                self._registered = False
            self._slots.clear()
            self._mapping_view = None
            self._mapping.close()
            if self._owns_mapping:
                self._mapping.unlink()
            self._mapping = None
