"""Native-Windows AMD multi-GPU transport experiments."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "D3D12AllReduce",
    "D3D12TensorBridge",
    "HostStagedGloo",
    "SharedMemoryAllReduce",
    "Win32PairBarrier",
]

_EXPORTS = {
    "D3D12AllReduce": (".d3d12_all_reduce", "D3D12AllReduce"),
    "D3D12TensorBridge": (".d3d12_transfer", "D3D12TensorBridge"),
    "HostStagedGloo": (".host_staged", "HostStagedGloo"),
    "SharedMemoryAllReduce": (".shared_memory_all_reduce", "SharedMemoryAllReduce"),
    "Win32PairBarrier": (".win32_barrier", "Win32PairBarrier"),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
