"""Benchmark simultaneous cross-process GPU exchange through D3D12 heaps."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
from queue import Empty
import time
import uuid

from windows_amd_vllm_multigpu.d3d12_shared import D3D12SharedBuffer
from windows_amd_vllm_multigpu.hip_runtime import HIP_MEMCPY_DEFAULT, HipRuntime


SIZE_BYTES = int(os.environ.get("WAVMG_D3D12_SIZE_BYTES", str(64 * 1024 * 1024)))
ITERATIONS = int(os.environ.get("WAVMG_D3D12_ITERATIONS", "100"))
WARMUP_ITERATIONS = int(os.environ.get("WAVMG_D3D12_WARMUP_ITERATIONS", "10"))
TIMEOUT_SECONDS = 180


def _rank_main(
    rank: int,
    base_name: str,
    creator_ready: list[mp.synchronize.Event],
    start_barrier: mp.synchronize.Barrier,
    results: mp.Queue,
) -> None:
    import torch

    record: dict[str, object] = {"rank": rank, "device": rank}
    own = None
    peer = None
    try:
        torch.cuda.set_device(rank)
        runtime = HipRuntime()
        own_name = f"{base_name}.rank{rank}"
        peer_name = f"{base_name}.rank{1 - rank}"
        own = D3D12SharedBuffer(own_name, SIZE_BYTES, rank, create=True)
        creator_ready[rank].set()
        if not creator_ready[1 - rank].wait(TIMEOUT_SECONDS):
            raise TimeoutError("peer did not create its D3D12 heap")
        peer = D3D12SharedBuffer(peer_name, SIZE_BYTES, rank, create=False)

        count = SIZE_BYTES // 4
        stream = torch.cuda.current_stream(rank)
        stream_pointer = int(stream.cuda_stream)
        outgoing_value = float(rank + 1)
        expected_value = float(2 - rank)
        outgoing = torch.full(
            (count,), outgoing_value, dtype=torch.float32, device=f"cuda:{rank}"
        )
        incoming = torch.empty_like(outgoing)

        def exchange(iteration: int) -> None:
            ready_value = (iteration * 2) + 1
            consumed_value = ready_value + 1
            runtime.memcpy_async(
                own.device_pointer,
                outgoing.data_ptr(),
                SIZE_BYTES,
                HIP_MEMCPY_DEFAULT,
                stream_pointer,
            )
            own.signal(ready_value, stream_pointer)
            peer.wait(ready_value, stream_pointer)
            runtime.memcpy_async(
                incoming.data_ptr(),
                peer.device_pointer,
                SIZE_BYTES,
                HIP_MEMCPY_DEFAULT,
                stream_pointer,
            )
            peer.signal(consumed_value, stream_pointer)
            own.wait(consumed_value, stream_pointer)

        for iteration in range(WARMUP_ITERATIONS):
            exchange(iteration)
        runtime.stream_synchronize(stream_pointer)
        start_barrier.wait(TIMEOUT_SECONDS)

        started = time.perf_counter()
        for iteration in range(WARMUP_ITERATIONS, WARMUP_ITERATIONS + ITERATIONS):
            exchange(iteration)
        runtime.stream_synchronize(stream_pointer)
        elapsed = time.perf_counter() - started
        correct = bool(torch.all(incoming == expected_value).item())
        record.update(
            {
                "opened": True,
                "correct": correct,
                "size_bytes": SIZE_BYTES,
                "iterations": ITERATIONS,
                "warmup_iterations": WARMUP_ITERATIONS,
                "elapsed_seconds": elapsed,
                "latency_ms": elapsed * 1000.0 / ITERATIONS,
                "logical_gb_s": (SIZE_BYTES * ITERATIONS) / elapsed / 1e9,
            }
        )
    except Exception as error:
        record.update(
            {
                "opened": False,
                "correct": False,
                "error": f"{type(error).__name__}: {error}",
            }
        )
        for event in creator_ready:
            event.set()
        try:
            start_barrier.abort()
        except Exception:
            pass
    finally:
        try:
            import torch

            torch.cuda.synchronize(rank)
        except Exception:
            pass
        for shared in (peer, own):
            if shared is not None:
                try:
                    shared.close()
                except Exception as error:
                    record.setdefault("close_errors", []).append(
                        f"{type(error).__name__}: {error}"
                    )
        results.put(record)


def main() -> int:
    if SIZE_BYTES <= 0 or SIZE_BYTES % 4:
        raise ValueError("WAVMG_D3D12_SIZE_BYTES must be a positive multiple of four")
    if ITERATIONS < 1 or WARMUP_ITERATIONS < 0:
        raise ValueError("iteration counts are invalid")
    context = mp.get_context("spawn")
    base_name = f"Local\\wavmg-benchmark-{uuid.uuid4().hex}"
    creator_ready = [context.Event(), context.Event()]
    start_barrier = context.Barrier(2)
    results = context.Queue()
    processes = [
        context.Process(
            target=_rank_main,
            args=(rank, base_name, creator_ready, start_barrier, results),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(TIMEOUT_SECONDS)
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(10)

    records = []
    for _ in processes:
        try:
            records.append(results.get(timeout=5))
        except Empty:
            break
    records.sort(key=lambda item: int(item["rank"]))
    summary = {
        "transport": "D3D12 cross-adapter L0/system-memory heap",
        "size_bytes": SIZE_BYTES,
        "iterations": ITERATIONS,
        "warmup_iterations": WARMUP_ITERATIONS,
        "ranks": records,
        "exit_codes": [process.exitcode for process in processes],
    }
    summary["passed"] = (
        len(records) == 2
        and all(bool(record.get("opened")) for record in records)
        and all(bool(record.get("correct")) for record in records)
        and all(code == 0 for code in summary["exit_codes"])
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
