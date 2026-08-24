"""ctypes-backed native Windows RCCL communicator for HIP tensors."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import torch
import torch.distributed as dist


NCCL_SUM = 0
_DTYPES = {
    torch.int8: 0,
    torch.uint8: 1,
    torch.int32: 2,
    torch.int64: 4,
    torch.float16: 6,
    torch.float32: 7,
    torch.float64: 8,
    torch.bfloat16: 9,
}


class NcclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_char * 128)]


def _enabled(value: str | None) -> bool:
    return (value or "").lower() in {"1", "true", "yes", "on"}


def rccl_requested() -> bool:
    return _enabled(os.environ.get("WAVMG_USE_RCCL"))


def _find_rccl() -> Path:
    configured = os.environ.get("WAVMG_RCCL_DLL")
    if configured:
        candidate = Path(configured).expanduser().resolve()
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"WAVMG_RCCL_DLL does not exist: {candidate}")

    candidates = [
        Path(__file__).resolve().parents[2] / "build" / "rccl-windows" / "rccl.dll",
        Path.cwd() / "build" / "rccl-windows" / "rccl.dll",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "rccl.dll was not found; build it with scripts\\configure-rccl-windows.ps1 "
        "and scripts\\build-rccl-windows.ps1, or set WAVMG_RCCL_DLL"
    )


class RcclCommunicator:
    """RCCL collectives launched on PyTorch's current HIP stream."""

    def __init__(
        self,
        group: dist.ProcessGroup | None,
        device: torch.device | str | int,
    ) -> None:
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized before RCCL")
        self.group = group
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        self.device = (
            torch.device("cuda", device)
            if isinstance(device, int)
            else torch.device(device)
        )
        if self.device.type != "cuda":
            raise ValueError(f"expected a HIP/CUDA-like device, got {self.device}")
        self._comm = ctypes.c_void_p()
        self._dll_directories: list[object] = []
        if self.world_size == 1:
            self.library = None
            return

        os.environ.setdefault("NCCL_RAS_ENABLE", "0")
        os.environ.setdefault("NCCL_SHM_DISABLE", "1")
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        os.environ.setdefault("NCCL_HOSTID", "windows-local")
        os.environ.setdefault("NCCL_COMM_BLOCKING", "1")
        os.environ.setdefault("NCCL_ALGO", "Ring")
        os.environ.setdefault("NCCL_PROTO", "Simple")

        torch.cuda.set_device(self.device)
        self.library = self._load_library(_find_rccl())
        descriptor: list[bytes | None] = [None]
        if self.rank == 0:
            unique_id = NcclUniqueId()
            self._check(self.library.ncclGetUniqueId(ctypes.byref(unique_id)), "ncclGetUniqueId")
            descriptor[0] = bytes(unique_id)
        global_source = dist.get_global_rank(group, 0) if group is not None else 0
        dist.broadcast_object_list(descriptor, src=global_source, group=group)
        if descriptor[0] is None:
            raise RuntimeError("rank 0 did not broadcast an RCCL unique ID")
        unique_id = NcclUniqueId.from_buffer_copy(descriptor[0])
        self._check(
            self.library.ncclCommInitRank(
                ctypes.byref(self._comm), self.world_size, unique_id, self.rank
            ),
            "ncclCommInitRank",
        )

    def _load_library(self, path: Path) -> ctypes.CDLL:
        site_packages = Path(torch.__file__).resolve().parent.parent
        for directory in (
            site_packages / "_rocm_sdk_devel" / "bin",
            Path(torch.__file__).resolve().parent / "lib",
            path.parent,
        ):
            if directory.is_dir():
                self._dll_directories.append(os.add_dll_directory(str(directory)))
        library = ctypes.CDLL(str(path))
        library.ncclGetUniqueId.argtypes = [ctypes.POINTER(NcclUniqueId)]
        library.ncclGetUniqueId.restype = ctypes.c_int
        library.ncclCommInitRank.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            NcclUniqueId,
            ctypes.c_int,
        ]
        library.ncclCommInitRank.restype = ctypes.c_int
        library.ncclCommDestroy.argtypes = [ctypes.c_void_p]
        library.ncclCommDestroy.restype = ctypes.c_int
        library.ncclGetErrorString.argtypes = [ctypes.c_int]
        library.ncclGetErrorString.restype = ctypes.c_char_p
        library.ncclAllReduce.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        library.ncclAllReduce.restype = ctypes.c_int
        library.ncclAllGather.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        library.ncclAllGather.restype = ctypes.c_int
        library.ncclReduceScatter.argtypes = library.ncclAllReduce.argtypes
        library.ncclReduceScatter.restype = ctypes.c_int
        library.ncclBroadcast.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        library.ncclBroadcast.restype = ctypes.c_int
        return library

    def _check(self, result: int, operation: str) -> None:
        if result == 0:
            return
        if self.library is None:
            message = "RCCL library is not initialized"
        else:
            raw = self.library.ncclGetErrorString(result)
            message = raw.decode("utf-8", errors="replace") if raw else "unknown"
        raise RuntimeError(f"{operation} failed: RCCL {result} ({message})")

    def _prepare(self, tensor: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        if tensor.device != self.device:
            raise ValueError(f"tensor is on {tensor.device}, communicator is {self.device}")
        data_type = _DTYPES.get(tensor.dtype)
        if data_type is None:
            raise TypeError(f"RCCL dtype is not supported: {tensor.dtype}")
        source = tensor if tensor.is_contiguous() else tensor.contiguous()
        stream = int(torch.cuda.current_stream(self.device).cuda_stream)
        return source, data_type, stream

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.world_size == 1:
            return tensor
        source, data_type, stream = self._prepare(tensor)
        output = torch.empty_like(source)
        self._check(
            self.library.ncclAllReduce(
                source.data_ptr(), output.data_ptr(), source.numel(), data_type,
                NCCL_SUM, self._comm, stream
            ),
            "ncclAllReduce",
        )
        return output

    def all_gather(self, tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if self.world_size == 1:
            return tensor
        source, data_type, stream = self._prepare(tensor)
        gathered = torch.empty(
            (self.world_size, *source.shape), dtype=source.dtype, device=source.device
        )
        self._check(
            self.library.ncclAllGather(
                source.data_ptr(), gathered.data_ptr(), source.numel(), data_type,
                self._comm, stream
            ),
            "ncclAllGather",
        )
        return torch.cat(list(gathered.unbind(0)), dim=dim)

    def reduce_scatter(self, tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if self.world_size == 1:
            return tensor
        if dim < 0:
            dim += tensor.dim()
        moved = tensor.movedim(dim, 0).contiguous()
        if moved.shape[0] % self.world_size:
            raise ValueError("scatter dimension must be divisible by world size")
        source, data_type, stream = self._prepare(moved)
        output = torch.empty(
            (moved.shape[0] // self.world_size, *moved.shape[1:]),
            dtype=moved.dtype,
            device=moved.device,
        )
        self._check(
            self.library.ncclReduceScatter(
                source.data_ptr(), output.data_ptr(), output.numel(), data_type,
                NCCL_SUM, self._comm, stream
            ),
            "ncclReduceScatter",
        )
        return output.movedim(0, dim).contiguous()

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        if self.world_size == 1:
            return tensor
        source, data_type, stream = self._prepare(tensor)
        output = torch.empty_like(source)
        self._check(
            self.library.ncclBroadcast(
                source.data_ptr(), output.data_ptr(), source.numel(), data_type,
                src, self._comm, stream
            ),
            "ncclBroadcast",
        )
        return output

    def destroy(self) -> None:
        if not self._comm.value or self.library is None:
            return
        torch.cuda.synchronize(self.device)
        self._check(self.library.ncclCommDestroy(self._comm), "ncclCommDestroy")
        self._comm = ctypes.c_void_p()
