"""ctypes wrapper for the Windows D3D12/HIP cross-adapter transport."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import torch

from .runtime_paths import rocm_bin_directories


class D3D12SharedBuffer:
    """One named cross-adapter heap and timeline fence mapped into HIP."""

    def __init__(
        self,
        name: str,
        size: int,
        device: int,
        *,
        create: bool,
        library_path: str | os.PathLike[str] | None = None,
    ) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        if library_path is None:
            configured = os.environ.get("WAVMG_D3D12_DLL")
            if configured:
                library_path = configured
            else:
                project_root = Path(__file__).resolve().parents[2]
                library_path = (
                    project_root / "build" / "native" / "wavmg_d3d12_v1.dll"
                )
        library_path = Path(library_path).resolve()
        if not library_path.is_file():
            raise FileNotFoundError(
                f"D3D12 transport DLL was not found at {library_path}; "
                "run scripts\\build-native.cmd"
            )

        self._dll_directories = [
            os.add_dll_directory(str(directory))
            for directory in rocm_bin_directories()
        ]
        self._dll_directories.append(os.add_dll_directory(str(library_path.parent)))
        self._library = ctypes.WinDLL(str(library_path))
        for function_name in ("wavmg_d3d12_create", "wavmg_d3d12_open"):
            function = getattr(self._library, function_name)
            function.argtypes = [
                ctypes.c_int,
                ctypes.c_wchar_p,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
            ]
            function.restype = ctypes.c_int
        for function_name in ("wavmg_d3d12_signal", "wavmg_d3d12_wait"):
            function = getattr(self._library, function_name)
            function.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint64,
                ctypes.c_void_p,
            ]
            function.restype = ctypes.c_int
        self._library.wavmg_d3d12_close.argtypes = [ctypes.c_void_p]
        self._library.wavmg_d3d12_close.restype = ctypes.c_int
        self._library.wavmg_d3d12_last_error.argtypes = []
        self._library.wavmg_d3d12_last_error.restype = ctypes.c_char_p

        self.name = name
        self.size = size
        self.device = device
        self._context = ctypes.c_void_p()
        pointer = ctypes.c_void_p()
        operation = (
            self._library.wavmg_d3d12_create
            if create
            else self._library.wavmg_d3d12_open
        )
        code = operation(
            device,
            name,
            size,
            ctypes.byref(self._context),
            ctypes.byref(pointer),
        )
        self._check(code, "create" if create else "open")
        if pointer.value is None:
            self.close()
            raise RuntimeError("D3D12 transport returned a null HIP pointer")
        self.device_pointer = int(pointer.value)

    def _check(self, code: int, operation: str) -> None:
        if code:
            raw_message = self._library.wavmg_d3d12_last_error()
            message = (
                raw_message.decode("utf-8", errors="replace")
                if raw_message
                else "unknown error"
            )
            raise RuntimeError(f"D3D12 transport {operation} failed: {message}")

    def signal(self, value: int, stream: int) -> None:
        self._check(
            self._library.wavmg_d3d12_signal(
                self._context, ctypes.c_uint64(value), ctypes.c_void_p(stream)
            ),
            "signal",
        )

    def wait(self, value: int, stream: int) -> None:
        self._check(
            self._library.wavmg_d3d12_wait(
                self._context, ctypes.c_uint64(value), ctypes.c_void_p(stream)
            ),
            "wait",
        )

    def close(self) -> None:
        if getattr(self, "_context", None) and self._context.value is not None:
            context = self._context
            self._context = ctypes.c_void_p()
            self._check(self._library.wavmg_d3d12_close(context), "close")

    def __enter__(self) -> "D3D12SharedBuffer":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
