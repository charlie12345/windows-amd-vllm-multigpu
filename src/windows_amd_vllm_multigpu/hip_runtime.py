"""Small ctypes binding for the HIP host-registration and copy APIs we use."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path


HIP_MEMCPY_HOST_TO_DEVICE = 1
HIP_MEMCPY_DEVICE_TO_HOST = 2
HIP_MEMCPY_DEFAULT = 4
HIP_HOST_REGISTER_MAPPED = 2
HIP_STREAM_WAIT_VALUE_GTE = 0
HIP_STREAM_NON_BLOCKING = 1
HIP_FULL_MASK = 0xFFFFFFFF
HIP_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE = 10013


class HipRuntime:
    def __init__(self) -> None:
        import torch

        site_packages = Path(torch.__file__).resolve().parent.parent
        runtime_dir = site_packages / "_rocm_sdk_devel" / "bin"
        runtime_path = runtime_dir / "amdhip64_7.dll"
        if not runtime_path.is_file():
            raise FileNotFoundError(f"HIP runtime was not found at {runtime_path}")
        self._dll_directory = os.add_dll_directory(str(runtime_dir))
        self.library = ctypes.WinDLL(str(runtime_path))
        self.library.hipHostRegister.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint,
        ]
        self.library.hipHostRegister.restype = ctypes.c_int
        self.library.hipHostUnregister.argtypes = [ctypes.c_void_p]
        self.library.hipHostUnregister.restype = ctypes.c_int
        self.library.hipHostGetDevicePointer.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_uint,
        ]
        self.library.hipHostGetDevicePointer.restype = ctypes.c_int
        self.library.hipMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self.library.hipMemcpy.restype = ctypes.c_int
        self.library.hipMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.library.hipMemcpyAsync.restype = ctypes.c_int
        self.library.hipStreamWriteValue32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint,
        ]
        self.library.hipStreamWriteValue32.restype = ctypes.c_int
        self.library.hipStreamWaitValue32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint,
            ctypes.c_uint32,
        ]
        self.library.hipStreamWaitValue32.restype = ctypes.c_int
        self.library.hipStreamSynchronize.argtypes = [ctypes.c_void_p]
        self.library.hipStreamSynchronize.restype = ctypes.c_int
        self.library.hipDeviceGetAttribute.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.library.hipDeviceGetAttribute.restype = ctypes.c_int
        self.library.hipGetErrorString.argtypes = [ctypes.c_int]
        self.library.hipGetErrorString.restype = ctypes.c_char_p

    def check(self, code: int, operation: str) -> None:
        if code:
            message = self.library.hipGetErrorString(code)
            decoded = message.decode("utf-8", errors="replace") if message else "unknown"
            raise RuntimeError(f"{operation} failed: HIP {code} ({decoded})")

    def host_register(self, pointer: int, size: int, flags: int = 0) -> None:
        self.check(
            self.library.hipHostRegister(ctypes.c_void_p(pointer), size, flags),
            "hipHostRegister",
        )

    def host_get_device_pointer(self, host_pointer: int) -> int:
        device_pointer = ctypes.c_void_p()
        self.check(
            self.library.hipHostGetDevicePointer(
                ctypes.byref(device_pointer), ctypes.c_void_p(host_pointer), 0
            ),
            "hipHostGetDevicePointer",
        )
        if device_pointer.value is None:
            raise RuntimeError("hipHostGetDevicePointer returned a null pointer")
        return int(device_pointer.value)

    def host_unregister(self, pointer: int) -> None:
        self.check(
            self.library.hipHostUnregister(ctypes.c_void_p(pointer)),
            "hipHostUnregister",
        )

    def memcpy(self, destination: int, source: int, size: int, kind: int) -> None:
        self.check(
            self.library.hipMemcpy(
                ctypes.c_void_p(destination),
                ctypes.c_void_p(source),
                size,
                kind,
            ),
            "hipMemcpy",
        )

    def memcpy_async(
        self,
        destination: int,
        source: int,
        size: int,
        kind: int,
        stream: int,
    ) -> None:
        self.check(
            self.library.hipMemcpyAsync(
                ctypes.c_void_p(destination),
                ctypes.c_void_p(source),
                size,
                kind,
                ctypes.c_void_p(stream),
            ),
            "hipMemcpyAsync",
        )

    def stream_write_value32(
        self, stream: int, device_pointer: int, value: int
    ) -> None:
        self.check(
            self.library.hipStreamWriteValue32(
                ctypes.c_void_p(stream),
                ctypes.c_void_p(device_pointer),
                ctypes.c_uint32(value),
                0,
            ),
            "hipStreamWriteValue32",
        )

    def stream_wait_value32(
        self, stream: int, device_pointer: int, value: int
    ) -> None:
        self.check(
            self.library.hipStreamWaitValue32(
                ctypes.c_void_p(stream),
                ctypes.c_void_p(device_pointer),
                ctypes.c_uint32(value),
                HIP_STREAM_WAIT_VALUE_GTE,
                HIP_FULL_MASK,
            ),
            "hipStreamWaitValue32",
        )

    def stream_synchronize(self, stream: int) -> None:
        self.check(
            self.library.hipStreamSynchronize(ctypes.c_void_p(stream)),
            "hipStreamSynchronize",
        )

    def can_use_stream_wait_value(self, device: int) -> bool:
        supported = ctypes.c_int()
        self.check(
            self.library.hipDeviceGetAttribute(
                ctypes.byref(supported),
                HIP_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE,
                device,
            ),
            "hipDeviceGetAttribute(CanUseStreamWaitValue)",
        )
        return bool(supported.value)
