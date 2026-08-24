"""Benchmark a fused GPU add that reads the peer rank's mapped host slot."""

from __future__ import annotations

import argparse
import ctypes
import json
import multiprocessing as mp
from multiprocessing import shared_memory
from queue import Empty
import time
import uuid


WORLD_SIZE = 2
TIMEOUT_SECONDS = 120
DEFAULT_SIZES = [4 * 1024, 64 * 1024, 1024 * 1024, 8 * 1024 * 1024]


def _iterations(size_bytes: int) -> int:
    if size_bytes <= 64 * 1024:
        return 100
    if size_bytes <= 1024 * 1024:
        return 30
    return 10


def _rank_main(
    rank: int,
    mapping_name: str,
    event_prefix: str,
    max_size_bytes: int,
    sizes: list[int],
    results: mp.Queue,
) -> None:
    import torch

    from windows_amd_vllm_multigpu.hip_runtime import (
        HIP_HOST_REGISTER_MAPPED,
        HIP_MEMCPY_DEVICE_TO_HOST,
        HipRuntime,
    )
    from windows_amd_vllm_multigpu.mapped_peer_kernel import MappedPeerKernel
    from windows_amd_vllm_multigpu.win32_barrier import Win32PairBarrier

    mapping = None
    mapping_view = None
    registered = False
    base_pointer = 0
    barrier = None
    slots = []
    record: dict[str, object] = {"rank": rank, "results": []}
    try:
        torch.set_num_threads(1)
        torch.cuda.set_device(rank)
        runtime = HipRuntime()
        kernel = MappedPeerKernel()
        mapping = shared_memory.SharedMemory(name=mapping_name)
        mapping_view = mapping.buf
        base_pointer = ctypes.addressof(ctypes.c_char.from_buffer(mapping_view))
        runtime.host_register(
            base_pointer, len(mapping_view), flags=HIP_HOST_REGISTER_MAPPED
        )
        registered = True
        mapped_device_pointer = runtime.host_get_device_pointer(base_pointer)
        barrier = Win32PairBarrier(event_prefix, rank, TIMEOUT_SECONDS * 1000)

        slots = [
            torch.frombuffer(
                mapping_view,
                dtype=torch.uint8,
                count=max_size_bytes,
                offset=slot_rank * max_size_bytes,
            )
            for slot_rank in range(WORLD_SIZE)
        ]

        for size_bytes in sizes:
            count = max(1, size_bytes // 4)
            iterations = _iterations(size_bytes)
            source = torch.full((count,), float(rank + 1), device=f"cuda:{rank}")
            destination = torch.empty_like(source)
            peer_slot_pointer = (
                mapped_device_pointer + (1 - rank) * max_size_bytes
            )

            def all_reduce() -> None:
                runtime.memcpy(
                    slots[rank].data_ptr(),
                    source.data_ptr(),
                    size_bytes,
                    HIP_MEMCPY_DEVICE_TO_HOST,
                )
                barrier.wait()
                kernel.add_float32(source, peer_slot_pointer, destination)
                barrier.wait()

            for _ in range(3):
                all_reduce()
            barrier.wait()
            started = time.perf_counter()
            for _ in range(iterations):
                all_reduce()
            barrier.wait()
            latency_ms = (time.perf_counter() - started) * 1000.0 / iterations
            correct = bool(torch.all(destination == 3.0).item())
            record["results"].append(
                {
                    "size_bytes": size_bytes,
                    "iterations": iterations,
                    "latency_ms": latency_ms,
                    "logical_mib_s": (size_bytes / (1024 * 1024))
                    / (latency_ms / 1000.0),
                    "correct": correct,
                }
            )
        record["mapped_device_pointer"] = mapped_device_pointer
        record["passed"] = all(item["correct"] for item in record["results"])
    except Exception as error:
        record["passed"] = False
        record["error"] = f"{type(error).__name__}: {error}"
    finally:
        if barrier is not None:
            barrier.close()
        if registered:
            try:
                runtime.host_unregister(base_pointer)
            except Exception as error:
                record["unregister_error"] = f"{type(error).__name__}: {error}"
        slots.clear()
        mapping_view = None
        if mapping is not None:
            mapping.close()
        results.put(record)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="*", type=int, default=DEFAULT_SIZES)
    args = parser.parse_args()
    if not args.sizes or any(size <= 0 or size % 4 for size in args.sizes):
        raise SystemExit("sizes must be positive multiples of four bytes")

    max_size = max(args.sizes)
    context = mp.get_context("spawn")
    mapping = shared_memory.SharedMemory(create=True, size=max_size * WORLD_SIZE)
    event_prefix = f"WAVMG_{uuid.uuid4().hex}"
    queue = context.Queue()
    processes = [
        context.Process(
            target=_rank_main,
            args=(rank, mapping.name, event_prefix, max_size, args.sizes, queue),
        )
        for rank in range(WORLD_SIZE)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(TIMEOUT_SECONDS + 30)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(5)

        records = []
        for _ in processes:
            try:
                records.append(queue.get(timeout=5))
            except Empty:
                break
        records.sort(key=lambda item: int(item["rank"]))
        summary = {
            "transport": "mapped peer-host read + fused HIP add",
            "world_size": WORLD_SIZE,
            "ranks": records,
            "exit_codes": [process.exitcode for process in processes],
        }
        summary["passed"] = (
            len(records) == WORLD_SIZE
            and all(bool(record.get("passed")) for record in records)
            and all(code == 0 for code in summary["exit_codes"])
        )
        print(json.dumps(summary, indent=2))
        return 0 if summary["passed"] else 1
    finally:
        mapping.close()
        mapping.unlink()


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
