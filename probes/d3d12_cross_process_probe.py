"""Validate bidirectional GPU-only payload movement across two processes."""

from __future__ import annotations

import json
import multiprocessing as mp
from queue import Empty
import uuid

from windows_amd_vllm_multigpu.d3d12_shared import D3D12SharedBuffer
from windows_amd_vllm_multigpu.hip_runtime import HIP_MEMCPY_DEFAULT, HipRuntime


SIZE_BYTES = 1024 * 1024
TIMEOUT_SECONDS = 45
FIRST_VALUE = 3.25
SECOND_VALUE = 7.5


def _rank_main(
    rank: int,
    name: str,
    creator_ready: mp.Event,
    opener_ready: mp.Event,
    results: mp.Queue,
) -> None:
    import torch

    record: dict[str, object] = {"rank": rank, "device": rank}
    shared = None
    try:
        torch.cuda.set_device(rank)
        runtime = HipRuntime()
        if rank == 0:
            shared = D3D12SharedBuffer(name, SIZE_BYTES, rank, create=True)
            creator_ready.set()
            if not opener_ready.wait(TIMEOUT_SECONDS):
                raise TimeoutError("peer did not open the D3D12 objects")
        else:
            if not creator_ready.wait(TIMEOUT_SECONDS):
                raise TimeoutError("creator did not publish the D3D12 objects")
            shared = D3D12SharedBuffer(name, SIZE_BYTES, rank, create=False)
            opener_ready.set()

        count = SIZE_BYTES // 4
        stream = torch.cuda.current_stream(rank)
        stream_pointer = int(stream.cuda_stream)
        if rank == 0:
            outgoing = torch.full(
                (count,), FIRST_VALUE, dtype=torch.float32, device="cuda:0"
            )
            incoming = torch.empty_like(outgoing)
            runtime.memcpy_async(
                shared.device_pointer,
                outgoing.data_ptr(),
                SIZE_BYTES,
                HIP_MEMCPY_DEFAULT,
                stream_pointer,
            )
            shared.signal(1, stream_pointer)
            shared.wait(2, stream_pointer)
            runtime.memcpy_async(
                incoming.data_ptr(),
                shared.device_pointer,
                SIZE_BYTES,
                HIP_MEMCPY_DEFAULT,
                stream_pointer,
            )
            runtime.stream_synchronize(stream_pointer)
            record["received_correct"] = bool(
                torch.all(incoming == SECOND_VALUE).item()
            )
        else:
            incoming = torch.empty(
                (count,), dtype=torch.float32, device="cuda:1"
            )
            outgoing = torch.full_like(incoming, SECOND_VALUE)
            shared.wait(1, stream_pointer)
            runtime.memcpy_async(
                incoming.data_ptr(),
                shared.device_pointer,
                SIZE_BYTES,
                HIP_MEMCPY_DEFAULT,
                stream_pointer,
            )
            runtime.memcpy_async(
                shared.device_pointer,
                outgoing.data_ptr(),
                SIZE_BYTES,
                HIP_MEMCPY_DEFAULT,
                stream_pointer,
            )
            shared.signal(2, stream_pointer)
            runtime.stream_synchronize(stream_pointer)
            record["received_correct"] = bool(
                torch.all(incoming == FIRST_VALUE).item()
            )
        record["mapped_pointer"] = shared.device_pointer
        record["opened"] = True
    except Exception as error:
        record["opened"] = False
        record["received_correct"] = False
        record["error"] = f"{type(error).__name__}: {error}"
        creator_ready.set()
        opener_ready.set()
    finally:
        if shared is not None:
            try:
                shared.close()
            except Exception as error:
                record["close_error"] = f"{type(error).__name__}: {error}"
        results.put(record)


def main() -> int:
    context = mp.get_context("spawn")
    name = f"Local\\wavmg-{uuid.uuid4().hex}"
    creator_ready = context.Event()
    opener_ready = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_rank_main,
            args=(rank, name, creator_ready, opener_ready, results),
        )
        for rank in range(2)
    ]
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
        "name": name,
        "size_bytes": SIZE_BYTES,
        "ranks": records,
        "exit_codes": [process.exitcode for process in processes],
    }
    summary["passed"] = (
        len(records) == 2
        and all(bool(record.get("opened")) for record in records)
        and all(bool(record.get("received_correct")) for record in records)
        and all(code == 0 for code in summary["exit_codes"])
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
