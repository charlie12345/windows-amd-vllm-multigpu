"""Probe which hipHostMalloc flag combinations work on native Windows HIP."""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path

import torch


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("HIP device is not available")

    rocm_bin = (
        Path(__file__).resolve().parents[1]
        / ".venv"
        / "Lib"
        / "site-packages"
        / "_rocm_sdk_devel"
        / "bin"
    )
    os.add_dll_directory(str(rocm_bin))
    hip = ctypes.WinDLL(str(rocm_bin / "amdhip64_7.dll"))
    hip.hipSetDevice.argtypes = [ctypes.c_int]
    hip.hipSetDevice.restype = ctypes.c_int
    hip.hipHostMalloc.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t,
        ctypes.c_uint,
    ]
    hip.hipHostMalloc.restype = ctypes.c_int
    hip.hipHostFree.argtypes = [ctypes.c_void_p]
    hip.hipHostFree.restype = ctypes.c_int
    hip.hipHostGetDevicePointer.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_uint,
    ]
    hip.hipHostGetDevicePointer.restype = ctypes.c_int

    set_device = hip.hipSetDevice(0)
    if set_device != 0:
        raise RuntimeError(f"hipSetDevice failed: {set_device}")

    flags = {
        "default": 0,
        "portable": 1,
        "mapped": 2,
        "portable_mapped": 3,
        "write_combined": 4,
        "mapped_write_combined": 6,
        "uncached": 0x10000000,
        "mapped_uncached": 0x10000002,
        "coherent": 0x40000000,
        "mapped_coherent": 0x40000002,
        "noncoherent": 0x80000000,
        "mapped_noncoherent": 0x80000002,
    }
    results: list[dict[str, object]] = []
    for name, value in flags.items():
        host_ptr = ctypes.c_void_p()
        alloc_result = hip.hipHostMalloc(ctypes.byref(host_ptr), 4096, value)
        device_ptr = ctypes.c_void_p()
        map_result = None
        free_result = None
        if alloc_result == 0:
            map_result = hip.hipHostGetDevicePointer(
                ctypes.byref(device_ptr), host_ptr, 0
            )
            free_result = hip.hipHostFree(host_ptr)
        results.append(
            {
                "name": name,
                "flags": value,
                "alloc_result": alloc_result,
                "host_pointer": host_ptr.value,
                "map_result": map_result,
                "device_pointer": device_ptr.value,
                "free_result": free_result,
            }
        )

    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
