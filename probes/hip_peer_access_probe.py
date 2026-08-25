"""Probe HIP peer capability separately from host-staged peer copies."""

from __future__ import annotations

import argparse
import ctypes
import json
import time
from collections.abc import Callable
from pathlib import Path

from windows_amd_vllm_multigpu.hip_runtime import (
    HIP_MEMCPY_DEVICE_TO_HOST,
    HIP_MEMCPY_HOST_TO_DEVICE,
    HipRuntime,
)

SIZE_BYTES = 4 * 1024 * 1024


def _status(operation: Callable[[], None]) -> dict[str, object]:
    try:
        operation()
        return {"succeeded": True, "hip_code": 0, "hip_error": "success"}
    except RuntimeError as error:
        message = str(error)
        code = None
        marker = "HIP "
        if marker in message:
            try:
                code = int(message.split(marker, 1)[1].split(" ", 1)[0])
            except ValueError:
                pass
        return {
            "succeeded": False,
            "hip_code": code,
            "hip_error": message,
        }


def _probe_direction(
    runtime: HipRuntime,
    source_device: int,
    destination_device: int,
    iterations: int,
) -> dict[str, object]:
    source_pointer = 0
    destination_pointer = 0
    peer_enabled = False
    record: dict[str, object] = {
        "source_device": source_device,
        "destination_device": destination_device,
        "can_access_peer": runtime.device_can_access_peer(
            source_device, destination_device
        ),
    }
    try:
        runtime.set_device(source_device)
        enable = _status(
            lambda: runtime.device_enable_peer_access(destination_device),
        )
        record["enable_peer_access"] = enable
        peer_enabled = bool(enable["succeeded"])

        runtime.set_device(source_device)
        source_pointer = runtime.malloc(SIZE_BYTES)
        runtime.set_device(destination_device)
        destination_pointer = runtime.malloc(SIZE_BYTES)

        expected = bytes(
            ((index * 29 + source_device * 17 + 11) & 0xFF)
            for index in range(SIZE_BYTES)
        )
        host_source = (ctypes.c_ubyte * SIZE_BYTES).from_buffer_copy(expected)
        runtime.set_device(source_device)
        runtime.memcpy(
            source_pointer,
            ctypes.addressof(host_source),
            SIZE_BYTES,
            HIP_MEMCPY_HOST_TO_DEVICE,
        )

        peer_copy = _status(
            lambda: runtime.memcpy_peer(
                destination_pointer,
                destination_device,
                source_pointer,
                source_device,
                SIZE_BYTES,
            ),
        )
        record["hip_memcpy_peer"] = peer_copy
        if peer_copy["succeeded"]:
            runtime.set_device(source_device)
            runtime.device_synchronize()
            runtime.set_device(destination_device)
            runtime.device_synchronize()
            start = time.perf_counter()
            for _ in range(iterations):
                runtime.memcpy_peer(
                    destination_pointer,
                    destination_device,
                    source_pointer,
                    source_device,
                    SIZE_BYTES,
                )
            runtime.set_device(source_device)
            runtime.device_synchronize()
            runtime.set_device(destination_device)
            runtime.device_synchronize()
            elapsed = time.perf_counter() - start
            peer_copy["iterations"] = iterations
            peer_copy["elapsed_seconds"] = elapsed
            peer_copy["gib_per_second"] = SIZE_BYTES * iterations / elapsed / (1024**3)
            host_destination = (ctypes.c_ubyte * SIZE_BYTES)()
            runtime.set_device(destination_device)
            runtime.memcpy(
                ctypes.addressof(host_destination),
                destination_pointer,
                SIZE_BYTES,
                HIP_MEMCPY_DEVICE_TO_HOST,
            )
            peer_copy["data_correct"] = bytes(host_destination) == expected
            peer_copy["evidence"] = (
                "peer_capability_enable_and_copy"
                if record["can_access_peer"] and peer_enabled
                else "copy_only_not_direct_p2p"
            )
    finally:
        if source_pointer:
            runtime.set_device(source_device)
            runtime.free(source_pointer)
        if destination_pointer:
            runtime.set_device(destination_device)
            runtime.free(destination_pointer)
        if peer_enabled:
            runtime.set_device(source_device)
            runtime.device_disable_peer_access(destination_device)
    return record


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe HIP peer capability separately from copy correctness."
    )
    parser.add_argument(
        "--runtime-dll",
        type=Path,
        help="Load this HIP runtime explicitly without importing PyTorch.",
    )
    parser.add_argument(
        "--dependency-dir",
        action="append",
        default=[],
        type=Path,
        help="Additional DLL search directory; may be specified more than once.",
    )
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--fail-closed-pal-peer-probe",
        action="store_true",
        help=(
            "Declare that the selected runtime has the repository's fail-closed "
            "PAL patch and GPU_FORCE_P2P_COMPAT=1. Only this mode can prove direct P2P."
        ),
    )
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    if arguments.iterations < 1:
        raise ValueError("--iterations must be at least 1")
    runtime = HipRuntime(arguments.runtime_dll, arguments.dependency_dir)
    device_count = runtime.get_device_count()
    result: dict[str, object] = {
        "runtime_dll": str(runtime.runtime_path),
        "hip_runtime_version": runtime.runtime_version(),
        "device_count": device_count,
        "size_bytes": SIZE_BYTES,
        "iterations": arguments.iterations,
        "directions": [],
    }
    if device_count < 2:
        result["passed"] = False
        result["error"] = "at least two HIP devices are required"
        print(json.dumps(result, indent=2))
        return 2

    directions = [
        _probe_direction(runtime, 0, 1, arguments.iterations),
        _probe_direction(runtime, 1, 0, arguments.iterations),
    ]
    result["directions"] = directions
    result["peer_capability_advertised"] = all(
        bool(direction["can_access_peer"]) for direction in directions
    )
    result["peer_enable_succeeded"] = all(
        bool(direction.get("enable_peer_access", {}).get("succeeded"))
        for direction in directions
    )
    result["peer_copy_correct"] = all(
        bool(direction.get("hip_memcpy_peer", {}).get("data_correct"))
        for direction in directions
    )
    result["fail_closed_pal_peer_probe"] = arguments.fail_closed_pal_peer_probe
    result["direct_p2p_proven"] = bool(
        arguments.fail_closed_pal_peer_probe
        and result["peer_capability_advertised"]
        and result["peer_enable_succeeded"]
        and result["peer_copy_correct"]
    )
    result["proof_note"] = (
        "The fail-closed runtime disables PAL host staging, so advertised peer "
        "access plus successful correct copies would prove peer-VRAM mappings."
        if arguments.fail_closed_pal_peer_probe
        else "Capability, enable, and copy correctness do not distinguish PAL peer "
        "VRAM mappings from its host-staging fallback; rerun through the "
        "fail-closed wrapper before claiming direct P2P."
    )
    result["candidate_passed"] = result["direct_p2p_proven"]
    print(json.dumps(result, indent=2))
    return 0 if result["candidate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
