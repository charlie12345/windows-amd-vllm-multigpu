"""Optional vLLM communicator adapter for the host-staged Gloo transport."""

from __future__ import annotations

import torch

from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)

from .host_staged import HostStagedGloo


class WindowsAmdHostStagedCommunicator(DeviceCommunicatorBase):
    """vLLM device communicator that never passes a GPU tensor to Gloo."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.use_all2all:
            raise NotImplementedError("expert-parallel all-to-all is not implemented")
        self.transport = HostStagedGloo(group=self.cpu_group, device=self.device)

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        return self.transport.all_reduce(input_)

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return self.transport.all_gather(input_, dim=dim)

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return self.transport.reduce_scatter(input_, dim=dim)

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
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

