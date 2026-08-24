"""vLLM adapter for the native Windows AMD transport and Gloo fallbacks."""

from __future__ import annotations

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

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        if self.d3d12 is not None:
            return self.d3d12.all_reduce(input_)
        if self.rccl is not None:
            return self.rccl.all_reduce(input_)
        if self.fast_all_reduce is not None:
            return self.fast_all_reduce.all_reduce(input_)
        return self.transport.all_reduce(input_)

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if self.rccl is not None:
            return self.rccl.all_gather(input_, dim=dim)
        return self.transport.all_gather(input_, dim=dim)

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if self.rccl is not None:
            return self.rccl.reduce_scatter(input_, dim=dim)
        return self.transport.reduce_scatter(input_, dim=dim)

    def gather(
        self, input_: torch.Tensor, dst: int = 0, dim: int = -1
    ) -> torch.Tensor | None:
        return self.transport.gather(input_, dst=dst, dim=dim)

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        if self.rccl is not None:
            return self.rccl.broadcast(tensor, src=src)
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
