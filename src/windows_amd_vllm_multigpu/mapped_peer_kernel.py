"""ctypes loader for the optional mapped-peer HIP reduction kernel."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import torch

from .runtime_paths import rocm_bin_directories


class MappedPeerKernel:
    def __init__(self, library_path: str | os.PathLike[str] | None = None) -> None:
        if library_path is None:
            configured = os.environ.get("WAVMG_HIP_DLL")
            if configured:
                library_path = configured
            else:
                project_root = Path(__file__).resolve().parents[2]
                library_path = project_root / "build" / "native" / "wavmg_hip_v1.dll"
        library_path = Path(library_path).resolve()
        if not library_path.is_file():
            raise FileNotFoundError(
                f"native HIP kernel was not found at {library_path}; "
                "run scripts\\build-native.cmd"
            )

        self._dll_directories = [
            os.add_dll_directory(str(directory)) for directory in rocm_bin_directories()
        ]
        self._dll_directories.append(os.add_dll_directory(str(library_path.parent)))
        self._library = ctypes.WinDLL(str(library_path))
        self._library.wavmg_add_f32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self._library.wavmg_add_f32.restype = ctypes.c_int
        for name in (
            "wavmg_add_f32_async",
            "wavmg_add_f16_async",
            "wavmg_add_bf16_async",
        ):
            function = getattr(self._library, name)
            function.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_void_p,
            ]
            function.restype = ctypes.c_int
        self._library.wavmg_copy_async.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ]
        self._library.wavmg_copy_async.restype = ctypes.c_int
        for name in (
            "wavmg_all_reduce_f32_async",
            "wavmg_all_reduce_f16_async",
            "wavmg_all_reduce_bf16_async",
        ):
            function = getattr(self._library, name)
            function.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.c_void_p,
            ]
            function.restype = ctypes.c_int

    def add_float32(
        self,
        local: torch.Tensor,
        peer_host_device_pointer: int,
        output: torch.Tensor,
    ) -> None:
        if local.dtype != torch.float32 or output.dtype != torch.float32:
            raise TypeError("the initial native kernel supports float32 only")
        if not local.is_contiguous() or not output.is_contiguous():
            raise ValueError("native kernel tensors must be contiguous")
        if local.numel() != output.numel():
            raise ValueError("native kernel input/output sizes differ")
        code = self._library.wavmg_add_f32(
            ctypes.c_void_p(local.data_ptr()),
            ctypes.c_void_p(peer_host_device_pointer),
            ctypes.c_void_p(output.data_ptr()),
            local.numel(),
        )
        if code:
            raise RuntimeError(f"wavmg_add_f32 failed with HIP error {code}")

    def add_async(
        self,
        local: torch.Tensor,
        peer_host_device_pointer: int,
        output: torch.Tensor,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Enqueue SUM(local, mapped peer slot) onto a PyTorch HIP stream."""
        if local.dtype != output.dtype:
            raise TypeError("native kernel input/output dtypes differ")
        if local.dtype == torch.float32:
            function = self._library.wavmg_add_f32_async
        elif local.dtype == torch.float16:
            function = self._library.wavmg_add_f16_async
        elif local.dtype == torch.bfloat16:
            function = self._library.wavmg_add_bf16_async
        else:
            raise TypeError(f"native kernel does not support {local.dtype}")
        if not local.is_contiguous() or not output.is_contiguous():
            raise ValueError("native kernel tensors must be contiguous")
        if local.numel() != output.numel():
            raise ValueError("native kernel input/output sizes differ")
        if stream is None:
            stream = torch.cuda.current_stream(local.device)
        code = function(
            ctypes.c_void_p(local.data_ptr()),
            ctypes.c_void_p(peer_host_device_pointer),
            ctypes.c_void_p(output.data_ptr()),
            local.numel(),
            ctypes.c_void_p(stream.cuda_stream),
        )
        if code:
            raise RuntimeError(f"mapped-peer add failed with HIP error {code}")

    def copy_async(
        self,
        source: torch.Tensor,
        destination_pointer: int,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Copy a contiguous tensor into mapped external memory on a HIP stream."""
        if not source.is_contiguous():
            raise ValueError("native copy source must be contiguous")
        if stream is None:
            stream = torch.cuda.current_stream(source.device)
        code = self._library.wavmg_copy_async(
            ctypes.c_void_p(source.data_ptr()),
            ctypes.c_void_p(destination_pointer),
            source.numel() * source.element_size(),
            ctypes.c_void_p(stream.cuda_stream),
        )
        if code:
            raise RuntimeError(f"mapped-peer copy failed with HIP error {code}")

    def all_reduce_async(
        self,
        local: torch.Tensor,
        local_host_pointer: int,
        peer_host_device_pointer: int,
        output: torch.Tensor,
        own_ready_device_pointer: int,
        peer_ready_device_pointer: int,
        own_consumed_device_pointer: int,
        peer_consumed_device_pointer: int,
        epoch: int,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Enqueue the complete stream-ordered two-rank SUM collective."""
        if local.dtype != output.dtype:
            raise TypeError("native collective input/output dtypes differ")
        if local.dtype == torch.float32:
            function = self._library.wavmg_all_reduce_f32_async
        elif local.dtype == torch.float16:
            function = self._library.wavmg_all_reduce_f16_async
        elif local.dtype == torch.bfloat16:
            function = self._library.wavmg_all_reduce_bf16_async
        else:
            raise TypeError(f"native collective does not support {local.dtype}")
        if not local.is_contiguous() or not output.is_contiguous():
            raise ValueError("native collective tensors must be contiguous")
        if local.numel() != output.numel():
            raise ValueError("native collective input/output sizes differ")
        if stream is None:
            stream = torch.cuda.current_stream(local.device)
        code = function(
            ctypes.c_void_p(local.data_ptr()),
            ctypes.c_void_p(local_host_pointer),
            ctypes.c_void_p(peer_host_device_pointer),
            ctypes.c_void_p(output.data_ptr()),
            local.numel(),
            ctypes.c_void_p(own_ready_device_pointer),
            ctypes.c_void_p(peer_ready_device_pointer),
            ctypes.c_void_p(own_consumed_device_pointer),
            ctypes.c_void_p(peer_consumed_device_pointer),
            ctypes.c_uint32(epoch),
            ctypes.c_void_p(stream.cuda_stream),
        )
        if code:
            raise RuntimeError(
                f"native all-reduce enqueue failed with HIP error {code}"
            )
