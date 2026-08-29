"""Test HIP-registering one Windows shared-memory mapping in two GPU ranks."""

from __future__ import annotations

import ctypes
import json
import multiprocessing as mp
from multiprocessing import shared_memory
from queue import Empty

from windows_amd_vllm_multigpu.hip_runtime import (
    HIP_HOST_REGISTER_MAPPED,
    HIP_MEMCPY_DEFAULT,
    HIP_MEMCPY_DEVICE_TO_HOST,
    HIP_MEMCPY_HOST_TO_DEVICE,
    HipRuntime,
)


WORLD_SIZE = 2
SIZE_BYTES = 1024 * 1024
TIMEOUT_SECONDS = 45


def _rank_main(
    rank: int,
    mapping_name: str,
    barrier: mp.Barrier,
    results: mp.Queue,
) -> None:
    import torch

    mapping = None
    mapping_view = None
    slots = []
    registered = False
    base_pointer = 0
    record: dict[str, object] = {"rank": rank}
    try:
        torch.cuda.set_device(rank)
        runtime = HipRuntime()
        mapping = shared_memory.SharedMemory(name=mapping_name)
        mapping_view = mapping.buf
        base_pointer = ctypes.addressof(ctypes.c_char.from_buffer(mapping_view))
        runtime.host_register(
            base_pointer, len(mapping_view), flags=HIP_HOST_REGISTER_MAPPED
        )
        registered = True
        mapped_device_pointer = runtime.host_get_device_pointer(base_pointer)

        count = SIZE_BYTES // 4
        slots = [
            torch.frombuffer(
                mapping_view,
                dtype=torch.float32,
                count=count,
                offset=slot_rank * SIZE_BYTES,
            )
            for slot_rank in range(WORLD_SIZE)
        ]
        source = torch.full((count,), float(rank + 1), device=f"cuda:{rank}")
        destination = torch.empty_like(source)
        local_sum = torch.empty((count,), dtype=torch.float32, pin_memory=True)

        runtime.memcpy(
            slots[rank].data_ptr(),
            source.data_ptr(),
            SIZE_BYTES,
            HIP_MEMCPY_DEVICE_TO_HOST,
        )
        barrier.wait(TIMEOUT_SECONDS)
        peer_copy = torch.empty_like(source)
        peer_slot_pointer = mapped_device_pointer + (1 - rank) * SIZE_BYTES
        runtime.memcpy(
            peer_copy.data_ptr(),
            peer_slot_pointer,
            SIZE_BYTES,
            HIP_MEMCPY_DEFAULT,
        )
        torch.add(slots[0], slots[1], out=local_sum)
        runtime.memcpy(
            destination.data_ptr(),
            local_sum.data_ptr(),
            SIZE_BYTES,
            HIP_MEMCPY_HOST_TO_DEVICE,
        )
        torch.cuda.synchronize(rank)
        record["correct"] = bool(torch.all(destination == 3.0).item())
        record["mapped_peer_copy_correct"] = bool(
            torch.all(peer_copy == float(2 if rank == 0 else 1)).item()
        )
        record["mapped_device_pointer"] = mapped_device_pointer
        record["mapping_pointer"] = base_pointer
        record["registered"] = True
        barrier.wait(TIMEOUT_SECONDS)
    except Exception as error:
        record["correct"] = False
        record["error"] = f"{type(error).__name__}: {error}"
    finally:
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
    context = mp.get_context("spawn")
    mapping = shared_memory.SharedMemory(create=True, size=SIZE_BYTES * WORLD_SIZE)
    barrier = context.Barrier(WORLD_SIZE)
    results = context.Queue()
    processes = [
        context.Process(
            target=_rank_main,
            args=(rank, mapping.name, barrier, results),
        )
        for rank in range(WORLD_SIZE)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(TIMEOUT_SECONDS + 15)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(5)

        records = []
        for _ in processes:
            try:
                records.append(results.get(timeout=5))
            except Empty:
                break
        records.sort(key=lambda item: int(item["rank"]))
        summary = {
            "mapping_name": mapping.name,
            "size_bytes_per_rank": SIZE_BYTES,
            "ranks": records,
            "exit_codes": [process.exitcode for process in processes],
        }
        summary["passed"] = (
            len(records) == WORLD_SIZE
            and all(bool(record.get("registered")) for record in records)
            and all(bool(record.get("correct")) for record in records)
            and all(bool(record.get("mapped_peer_copy_correct")) for record in records)
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
