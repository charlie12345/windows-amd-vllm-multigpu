"""vLLM adapter for the native Windows AMD transport and Gloo fallbacks."""

from __future__ import annotations

import os

import torch

from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)

from .d3d12_all_reduce import D3D12AllReduce, d3d12_requested
from .host_staged import HostStagedGloo
from .rccl import RcclCommunicator, rccl_requested
from .shared_memory_all_reduce import SharedMemoryAllReduce


class WindowsAmdMultiGpuCommunicator(DeviceCommunicatorBase):
    """vLLM communicator that never passes a GPU tensor directly to Gloo."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.use_all2all:
            raise NotImplementedError("expert-parallel all-to-all is not implemented")
        self.transport = HostStagedGloo(group=self.cpu_group, device=self.device)
        self.d3d12 = (
            D3D12AllReduce(group=self.cpu_group, device=self.device)
            if self.world_size == 2 and d3d12_requested()
            else None
        )
        self.rccl = (
            RcclCommunicator(group=self.cpu_group, device=self.device)
            if rccl_requested()
            else None
        )
        self.fast_all_reduce = (
            SharedMemoryAllReduce(group=self.cpu_group, device=self.device)
            if self.world_size == 2 and self.rccl is None and self.d3d12 is None
            else None
        )
        self._trace_enabled = os.environ.get("WAVMG_TRACE_COLLECTIVES", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._traced_operations: set[str] = set()
        if self._trace_enabled:
            if self.d3d12 is not None:
                all_reduce_backend = (
                    "hybrid(d3d12="
                    f"{self.d3d12.min_size_bytes}..{self.d3d12.max_size_bytes},"
                    "rccl=fallback)"
                )
            elif self.rccl is not None:
                all_reduce_backend = "rccl"
            elif self.fast_all_reduce is not None:
                all_reduce_backend = "mapped-host"
            else:
                all_reduce_backend = "gloo-host"
            other_backend = "rccl" if self.rccl is not None else "gloo-host"
            print(
                "WAVMG_TRANSPORT "
                f"rank={self.rank_in_group} world_size={self.world_size} "
                f"all_reduce={all_reduce_backend} "
                f"other_collectives={other_backend}",
                flush=True,
            )

    def _trace_once(
        self,
        operation: str,
        backend: str,
        tensor: torch.Tensor | None = None,
    ) -> None:
        size_bytes = None
        dtype = None
        shape = None
        if tensor is not None:
            size_bytes = tensor.numel() * tensor.element_size()
            dtype = tensor.dtype
            shape = tuple(tensor.shape)
        trace_key = f"{operation}:{backend}:{dtype}:{size_bytes}:{shape}"
        if not self._trace_enabled or trace_key in self._traced_operations:
            return
        self._traced_operations.add(trace_key)
        details = ""
        if tensor is not None:
            details = f" dtype={dtype} bytes={size_bytes} shape={shape}"
        print(
            f"WAVMG_COLLECTIVE rank={self.rank_in_group} "
            f"operation={operation} backend={backend}{details}",
            flush=True,
        )

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        if self.d3d12 is not None and self.d3d12.can_handle(input_):
            self._trace_once("all_reduce", "d3d12", input_)
            return self.d3d12.all_reduce(input_)
        if self.rccl is not None:
            backend = "rccl-d3d12-fallback" if self.d3d12 is not None else "rccl"
            self._trace_once("all_reduce", backend, input_)
            return self.rccl.all_reduce(input_)
        if self.fast_all_reduce is not None:
            self._trace_once("all_reduce", "mapped-host", input_)
            return self.fast_all_reduce.all_reduce(input_)
        self._trace_once("all_reduce", "gloo-host", input_)
        return self.transport.all_reduce(input_)

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if self.rccl is not None:
            self._trace_once("all_gather", "rccl")
            return self.rccl.all_gather(input_, dim=dim)
        self._trace_once("all_gather", "gloo-host")
        return self.transport.all_gather(input_, dim=dim)

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if self.rccl is not None:
            self._trace_once("reduce_scatter", "rccl")
            return self.rccl.reduce_scatter(input_, dim=dim)
        self._trace_once("reduce_scatter", "gloo-host")
        return self.transport.reduce_scatter(input_, dim=dim)

    def gather(
        self, input_: torch.Tensor, dst: int = 0, dim: int = -1
    ) -> torch.Tensor | None:
        return self.transport.gather(input_, dst=dst, dim=dim)

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        if self.rccl is not None:
            self._trace_once("broadcast", "rccl")
            return self.rccl.broadcast(tensor, src=src)
        self._trace_once("broadcast", "gloo-host")
        return self.transport.broadcast(tensor, src=src)

    def send(self, tensor: torch.Tensor, dst: int | None = None) -> None:
        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size
        self.transport.send(tensor, dst=dst)

    def recv(
        self,
        size: torch.Size,
        dtype: torch.dtype,
        src: int | None = None,
    ) -> torch.Tensor:
        if src is None:
            src = (self.rank_in_group - 1) % self.world_size
        return self.transport.recv(size, dtype, src=src)

    def destroy(self) -> None:
        if self.d3d12 is not None:
            self.d3d12.destroy()
        if self.rccl is not None:
            self.rccl.destroy()
        if self.fast_all_reduce is not None:
            self.fast_all_reduce.destroy()
