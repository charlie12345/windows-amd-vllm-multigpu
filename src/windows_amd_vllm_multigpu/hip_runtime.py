"""Small ctypes binding for the HIP host-registration and copy APIs we use."""

from __future__ import annotations

import ctypes
import os
from collections.abc import Iterable
from pathlib import Path

HIP_MEMCPY_HOST_TO_DEVICE = 1
HIP_MEMCPY_DEVICE_TO_HOST = 2
HIP_MEMCPY_DEFAULT = 4
HIP_HOST_REGISTER_MAPPED = 2
HIP_STREAM_WAIT_VALUE_GTE = 0
HIP_STREAM_NON_BLOCKING = 1
HIP_FULL_MASK = 0xFFFFFFFF
HIP_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE = 10013
HIP_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 1


class HipIpcMemHandle(ctypes.Structure):
    """ABI-compatible storage for hipIpcMemHandle_t."""

    _fields_ = [("reserved", ctypes.c_char * 64)]


class HipRuntime:
    def __init__(
        self,
        runtime_path: str | Path | None = None,
        dependency_dirs: Iterable[str | Path] = (),
    ) -> None:
        if runtime_path is None:
            import torch

            site_packages = Path(torch.__file__).resolve().parent.parent
            resolved_runtime_path = (
                site_packages / "_rocm_sdk_devel" / "bin" / "amdhip64_7.dll"
            )
        else:
            resolved_runtime_path = Path(runtime_path).expanduser().resolve()

        if not resolved_runtime_path.is_file():
            raise FileNotFoundError(
                f"HIP runtime was not found at {resolved_runtime_path}"
            )

        dll_directories = [resolved_runtime_path.parent]
        dll_directories.extend(
            Path(directory).expanduser().resolve() for directory in dependency_dirs
        )
        self._dll_directories = []
        for directory in dict.fromkeys(dll_directories):
            if not directory.is_dir():
                raise FileNotFoundError(
                    f"DLL dependency directory was not found: {directory}"
                )
            self._dll_directories.append(os.add_dll_directory(str(directory)))

        self.runtime_path = resolved_runtime_path
        self.library = ctypes.WinDLL(str(resolved_runtime_path))
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
        self.library.hipDeviceSynchronize.argtypes = []
        self.library.hipDeviceSynchronize.restype = ctypes.c_int
        self.library.hipDeviceGetAttribute.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.library.hipDeviceGetAttribute.restype = ctypes.c_int
        self.library.hipGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self.library.hipGetDeviceCount.restype = ctypes.c_int
        self.library.hipDeviceCanAccessPeer.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.library.hipDeviceCanAccessPeer.restype = ctypes.c_int
        self.library.hipDeviceEnablePeerAccess.argtypes = [
            ctypes.c_int,
            ctypes.c_uint,
        ]
        self.library.hipDeviceEnablePeerAccess.restype = ctypes.c_int
        self.library.hipDeviceDisablePeerAccess.argtypes = [ctypes.c_int]
        self.library.hipDeviceDisablePeerAccess.restype = ctypes.c_int
        self.library.hipSetDevice.argtypes = [ctypes.c_int]
        self.library.hipSetDevice.restype = ctypes.c_int
        self.library.hipMalloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
        ]
        self.library.hipMalloc.restype = ctypes.c_int
        self.library.hipFree.argtypes = [ctypes.c_void_p]
        self.library.hipFree.restype = ctypes.c_int
        self.library.hipMemcpyPeer.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_size_t,
        ]
        self.library.hipMemcpyPeer.restype = ctypes.c_int
        self.library.hipMemcpyPeerAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ]
        self.library.hipMemcpyPeerAsync.restype = ctypes.c_int
        self.library.hipIpcGetMemHandle.argtypes = [
            ctypes.POINTER(HipIpcMemHandle),
            ctypes.c_void_p,
        ]
        self.library.hipIpcGetMemHandle.restype = ctypes.c_int
        self.library.hipIpcOpenMemHandle.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            HipIpcMemHandle,
            ctypes.c_uint,
        ]
        self.library.hipIpcOpenMemHandle.restype = ctypes.c_int
        self.library.hipIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
        self.library.hipIpcCloseMemHandle.restype = ctypes.c_int
        self.library.hipGetErrorString.argtypes = [ctypes.c_int]
        self.library.hipGetErrorString.restype = ctypes.c_char_p
        self.library.hipRuntimeGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self.library.hipRuntimeGetVersion.restype = ctypes.c_int

    def check(self, code: int, operation: str) -> None:
        if code:
            raise RuntimeError(
                f"{operation} failed: HIP {code} ({self.error_string(code)})"
            )

    def error_string(self, code: int) -> str:
        message = self.library.hipGetErrorString(code)
        return message.decode("utf-8", errors="replace") if message else "unknown"

    def runtime_version(self) -> int:
        version = ctypes.c_int()
        self.check(
            self.library.hipRuntimeGetVersion(ctypes.byref(version)),
            "hipRuntimeGetVersion",
        )
        return int(version.value)

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

    def stream_wait_value32(self, stream: int, device_pointer: int, value: int) -> None:
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

    def device_synchronize(self) -> None:
        self.check(self.library.hipDeviceSynchronize(), "hipDeviceSynchronize")

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

    def get_device_count(self) -> int:
        count = ctypes.c_int()
        self.check(
            self.library.hipGetDeviceCount(ctypes.byref(count)),
            "hipGetDeviceCount",
        )
        return int(count.value)

    def device_can_access_peer(self, device: int, peer_device: int) -> bool:
        supported = ctypes.c_int()
        self.check(
            self.library.hipDeviceCanAccessPeer(
                ctypes.byref(supported), device, peer_device
            ),
            "hipDeviceCanAccessPeer",
        )
        return bool(supported.value)

    def device_enable_peer_access(self, peer_device: int) -> None:
        self.check(
            self.library.hipDeviceEnablePeerAccess(peer_device, 0),
            "hipDeviceEnablePeerAccess",
        )

    def device_disable_peer_access(self, peer_device: int) -> None:
        self.check(
            self.library.hipDeviceDisablePeerAccess(peer_device),
            "hipDeviceDisablePeerAccess",
        )

    def set_device(self, device: int) -> None:
        self.check(self.library.hipSetDevice(device), "hipSetDevice")

    def malloc(self, size: int) -> int:
        pointer = ctypes.c_void_p()
        self.check(self.library.hipMalloc(ctypes.byref(pointer), size), "hipMalloc")
        if pointer.value is None:
            raise RuntimeError("hipMalloc returned a null pointer")
        return int(pointer.value)

    def free(self, pointer: int) -> None:
        self.check(self.library.hipFree(ctypes.c_void_p(pointer)), "hipFree")

    def memcpy_peer(
        self,
        destination: int,
        destination_device: int,
        source: int,
        source_device: int,
        size: int,
    ) -> None:
        self.check(
            self.library.hipMemcpyPeer(
                ctypes.c_void_p(destination),
                destination_device,
                ctypes.c_void_p(source),
                source_device,
                size,
            ),
            "hipMemcpyPeer",
        )

    def memcpy_peer_async(
        self,
        destination: int,
        destination_device: int,
        source: int,
        source_device: int,
        size: int,
        stream: int,
    ) -> None:
        self.check(
            self.library.hipMemcpyPeerAsync(
                ctypes.c_void_p(destination),
                destination_device,
                ctypes.c_void_p(source),
                source_device,
                size,
                ctypes.c_void_p(stream),
            ),
            "hipMemcpyPeerAsync",
        )

    def ipc_get_mem_handle(self, device_pointer: int) -> bytes:
        handle = HipIpcMemHandle()
        self.check(
            self.library.hipIpcGetMemHandle(
                ctypes.byref(handle), ctypes.c_void_p(device_pointer)
            ),
            "hipIpcGetMemHandle",
        )
        return bytes(handle)

    def ipc_open_mem_handle(self, handle_bytes: bytes) -> int:
        if len(handle_bytes) != ctypes.sizeof(HipIpcMemHandle):
            raise ValueError(
                f"HIP IPC handle must be {ctypes.sizeof(HipIpcMemHandle)} bytes"
            )
        handle = HipIpcMemHandle.from_buffer_copy(handle_bytes)
        pointer = ctypes.c_void_p()
        self.check(
            self.library.hipIpcOpenMemHandle(
                ctypes.byref(pointer),
                handle,
                HIP_IPC_MEM_LAZY_ENABLE_PEER_ACCESS,
            ),
            "hipIpcOpenMemHandle",
        )
        if pointer.value is None:
            raise RuntimeError("hipIpcOpenMemHandle returned a null pointer")
        return int(pointer.value)

    def ipc_close_mem_handle(self, device_pointer: int) -> None:
        self.check(
            self.library.hipIpcCloseMemHandle(ctypes.c_void_p(device_pointer)),
            "hipIpcCloseMemHandle",
        )
