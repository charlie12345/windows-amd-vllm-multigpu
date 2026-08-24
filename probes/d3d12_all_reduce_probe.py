"""Validate and benchmark D3D12 cross-adapter SUM on PyTorch HIP streams."""

from __future__ import annotations

from datetime import timedelta
import json
import multiprocessing as mp
from queue import Empty
import socket
import time


WORLD_SIZE = 2
TIMEOUT_SECONDS = 180
SIZES = [4 * 1024, 64 * 1024, 1024 * 1024, 8 * 1024 * 1024, 64 * 1024 * 1024]
DTYPES = ("float16", "float32", "bfloat16")


def _unused_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _iterations(size_bytes: int) -> int:
    if size_bytes <= 64 * 1024:
        return 200
    if size_bytes <= 1024 * 1024:
        return 100
    if size_bytes <= 8 * 1024 * 1024:
        return 50
    return 20


def _rank_main(rank: int, port: int, results: mp.Queue) -> None:
    import torch
    import torch.distributed as dist

    from windows_amd_vllm_multigpu.d3d12_all_reduce import D3D12AllReduce

    record: dict[str, object] = {"rank": rank, "device": rank, "results": []}
    communicator = None
    try:
        torch.set_num_threads(1)
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=WORLD_SIZE,
            timeout=timedelta(seconds=TIMEOUT_SECONDS),
        )
        communicator = D3D12AllReduce(
            group=None, device=rank, max_size_bytes=max(SIZES)
        )
        dtypes = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        test_stream = torch.cuda.Stream(device=rank)
        for dtype_name in DTYPES:
            dtype = dtypes[dtype_name]
            element_size = torch.empty((), dtype=dtype).element_size()
            for size_bytes in SIZES:
                count = size_bytes // element_size
                iterations = _iterations(size_bytes)
                source = torch.full(
                    (count,), float(rank + 1), dtype=dtype, device=f"cuda:{rank}"
                )
                # The fill is on PyTorch's default stream; establish visibility
                # before deliberately exercising the communicator on another.
                torch.cuda.synchronize(rank)
                with torch.cuda.stream(test_stream):
                    for _ in range(10):
                        output = communicator.all_reduce(source)
                test_stream.synchronize()
                dist.barrier()
                started = time.perf_counter()
                with torch.cuda.stream(test_stream):
                    for _ in range(iterations):
                        output = communicator.all_reduce(source)
                test_stream.synchronize()
                elapsed = time.perf_counter() - started
                correct = bool(torch.all(output.float() == 3.0).item())
                first_value = float(output[0].float().item())
                record["results"].append(
                    {
                        "dtype": dtype_name,
                        "size_bytes": size_bytes,
                        "iterations": iterations,
                        "latency_ms": elapsed * 1000.0 / iterations,
                        "logical_gb_s": (size_bytes * iterations) / elapsed / 1e9,
                        "first_value": first_value,
                        "correct": correct,
                    }
                )
        record["passed"] = all(
            bool(item["correct"]) for item in record["results"]
        )
    except Exception as error:
        record["passed"] = False
        record["error"] = f"{type(error).__name__}: {error}"
    finally:
        if communicator is not None:
            try:
                communicator.destroy()
            except Exception as error:
                record["destroy_error"] = f"{type(error).__name__}: {error}"
                record["passed"] = False
        if "dist" in locals() and dist.is_initialized():
            dist.destroy_process_group()
        results.put(record)


def main() -> int:
    context = mp.get_context("spawn")
    port = _unused_local_port()
    results = context.Queue()
    processes = [
        context.Process(target=_rank_main, args=(rank, port, results))
        for rank in range(WORLD_SIZE)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(TIMEOUT_SECONDS + 30)
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
        "transport": "D3D12 cross-adapter fused SUM",
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


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
