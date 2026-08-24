"""Correctness-first GPU collectives staged through pinned CPU memory."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class _BufferKey:
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype


class HostStagedGloo:
    """Run collectives for one GPU per rank through a Gloo CPU process group.

    Direct Gloo collectives on Windows GPU tensors currently terminate the
    process inside ProcessGroupGloo. This class makes the host staging explicit
    so a GPU tensor is never handed to Gloo.
    """

    def __init__(
        self,
        group: dist.ProcessGroup | None = None,
        device: torch.device | str | int | None = None,
    ) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized first")
        if not dist.is_gloo_available():
            raise RuntimeError("the PyTorch build does not provide Gloo")

        self.group = group
        backend = str(dist.get_backend(group)).lower()
        if "gloo" not in backend:
            raise ValueError(f"HostStagedGloo requires a Gloo group, got {backend!r}")

        if device is None:
            device = torch.cuda.current_device()
        if isinstance(device, int):
            device = torch.device("cuda", device)
        else:
            device = torch.device(device)
        if device.type != "cuda":
            raise ValueError(f"expected a HIP/CUDA-like device, got {device}")

        self.device = device
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        self._host_buffers: dict[_BufferKey, torch.Tensor] = {}

    def _stage_for(self, tensor: torch.Tensor) -> torch.Tensor:
        key = _BufferKey(tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)
        stage = self._host_buffers.get(key)
        if stage is None:
            stage = torch.empty_strided(
                tensor.shape,
                tensor.stride(),
                dtype=tensor.dtype,
                device="cpu",
                pin_memory=True,
            )
            self._host_buffers[key] = stage
        return stage

    @staticmethod
    def _gpu_from_host(host: torch.Tensor, device: torch.device) -> torch.Tensor:
        output = torch.empty_strided(
            host.shape, host.stride(), dtype=host.dtype, device=device
        )
        output.copy_(host, non_blocking=False)
        return output

    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: dist.ReduceOp = dist.ReduceOp.SUM,
    ) -> torch.Tensor:
        """Return the all-reduced value on the input tensor's GPU."""
        if tensor.device != self.device:
            raise ValueError(f"tensor is on {tensor.device}, communicator is {self.device}")
        if self.world_size == 1:
            return tensor

        stage = self._stage_for(tensor)
        stage.copy_(tensor, non_blocking=False)
        dist.all_reduce(stage, op=op, group=self.group)
        return self._gpu_from_host(stage, tensor.device)

    def all_gather(self, tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Gather equal tensors from all ranks and concatenate along ``dim``."""
        if self.world_size == 1:
            return tensor
        stage = self._stage_for(tensor)
        stage.copy_(tensor, non_blocking=False)
        gathered = [torch.empty_like(stage) for _ in range(self.world_size)]
        dist.all_gather(gathered, stage, group=self.group)
        host_output = torch.cat(gathered, dim=dim)
        return self._gpu_from_host(host_output, tensor.device)

    def reduce_scatter(
        self,
        tensor: torch.Tensor,
        dim: int = -1,
        op: dist.ReduceOp = dist.ReduceOp.SUM,
    ) -> torch.Tensor:
        """Reduce an equal-size tensor and scatter chunks along ``dim``."""
        if self.world_size == 1:
            return tensor
        if dim < 0:
            dim += tensor.dim()
        moved = tensor.movedim(dim, 0).contiguous()
        if moved.shape[0] % self.world_size:
            raise ValueError("scatter dimension must be divisible by world size")
        stage = self._stage_for(moved)
        stage.copy_(moved, non_blocking=False)
        host_output = torch.empty(
            (moved.shape[0] // self.world_size, *moved.shape[1:]),
            dtype=moved.dtype,
            device="cpu",
            pin_memory=True,
        )
        dist.reduce_scatter_tensor(host_output, stage, op=op, group=self.group)
        output = self._gpu_from_host(host_output, tensor.device)
        return output.movedim(0, dim).contiguous()

    def gather(
        self, tensor: torch.Tensor, dst: int = 0, dim: int = -1
    ) -> torch.Tensor | None:
        """Gather equal GPU tensors to one group-local destination rank."""
        if self.world_size == 1:
            return tensor
        stage = self._stage_for(tensor)
        stage.copy_(tensor, non_blocking=False)
        global_dst = dist.get_global_rank(self.group, dst) if self.group else dst
        gathered = (
            [torch.empty_like(stage) for _ in range(self.world_size)]
            if self.rank == dst
            else None
        )
        dist.gather(stage, gather_list=gathered, dst=global_dst, group=self.group)
        if gathered is None:
            return None
        return self._gpu_from_host(torch.cat(gathered, dim=dim), tensor.device)

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Broadcast a GPU tensor from the group-local source rank."""
        if self.world_size == 1:
            return tensor
        stage = self._stage_for(tensor)
        stage.copy_(tensor, non_blocking=False)
        global_src = dist.get_global_rank(self.group, src) if self.group else src
        dist.broadcast(stage, src=global_src, group=self.group)
        tensor.copy_(stage, non_blocking=False)
        return tensor

    def send(self, tensor: torch.Tensor, dst: int) -> None:
        """Send a GPU tensor to a group-local destination rank."""
        stage = self._stage_for(tensor)
        stage.copy_(tensor, non_blocking=False)
        global_dst = dist.get_global_rank(self.group, dst) if self.group else dst
        dist.send(stage, dst=global_dst, group=self.group)

    def recv(
        self,
        shape: torch.Size | tuple[int, ...],
        dtype: torch.dtype,
        src: int,
    ) -> torch.Tensor:
        """Receive through Gloo CPU memory and return a tensor on this rank's GPU."""
        stage = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
        global_src = dist.get_global_rank(self.group, src) if self.group else src
        dist.recv(stage, src=global_src, group=self.group)
        return self._gpu_from_host(stage, self.device)

    def barrier(self) -> None:
        dist.barrier(group=self.group)
