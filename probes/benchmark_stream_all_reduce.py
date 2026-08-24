"""Benchmark the stream-ordered two-rank mapped-memory all-reduce."""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
import multiprocessing as mp
from queue import Empty
import socket
import time


WORLD_SIZE = 2
TIMEOUT_SECONDS = 120
DEFAULT_SIZES = [4 * 1024, 8 * 1024, 64 * 1024, 1024 * 1024, 8 * 1024 * 1024]
DTYPE_NAMES = ("float32", "float16", "bfloat16")


def _unused_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _iterations(size_bytes: int) -> int:
    if size_bytes <= 64 * 1024:
        return 200
    if size_bytes <= 1024 * 1024:
        return 50
    return 15


def _rank_main(
    rank: int,
    port: int,
    sizes: list[int],
    dtype_names: tuple[str, ...],
    results: mp.Queue,
) -> None:
    import torch
    import torch.distributed as dist

    from windows_amd_vllm_multigpu import SharedMemoryAllReduce

    record: dict[str, object] = {"rank": rank, "results": []}
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
        communicator = SharedMemoryAllReduce(
            group=None, device=rank, max_size_bytes=max(sizes)
        )
        dtypes = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        record["stream_path_enabled"] = communicator._use_stream_path

        for dtype_name in dtype_names:
            dtype = dtypes[dtype_name]
            element_size = torch.empty((), dtype=dtype).element_size()
            for size_bytes in sizes:
                if size_bytes % element_size:
                    raise ValueError(f"{size_bytes} is not aligned for {dtype_name}")
                count = size_bytes // element_size
                iterations = _iterations(size_bytes)
                source = torch.full(
                    (count,), float(rank + 1), dtype=dtype, device=f"cuda:{rank}"
                )
                destination = source
                for _ in range(5):
                    destination = communicator.all_reduce(source)
                torch.cuda.synchronize(rank)
                dist.barrier()
                started = time.perf_counter()
                for _ in range(iterations):
                    destination = communicator.all_reduce(source)
                torch.cuda.synchronize(rank)
                latency_ms = (time.perf_counter() - started) * 1000.0 / iterations
                correct = bool(torch.all(destination.float() == 3.0).cpu().item())
                record["results"].append(
                    {
                        "dtype": dtype_name,
                        "size_bytes": size_bytes,
                        "iterations": iterations,
                        "latency_ms": latency_ms,
                        "logical_mib_s": (size_bytes / (1024 * 1024))
                        / (latency_ms / 1000.0),
                        "correct": correct,
                    }
                )
        record["passed"] = bool(record["stream_path_enabled"]) and all(
            item["correct"] for item in record["results"]
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="*", type=int, default=DEFAULT_SIZES)
    parser.add_argument("--dtypes", nargs="*", choices=DTYPE_NAMES, default=DTYPE_NAMES)
    args = parser.parse_args()
    if not args.sizes or any(size <= 0 for size in args.sizes):
        raise SystemExit("sizes must be positive")

    context = mp.get_context("spawn")
    results = context.Queue()
    port = _unused_local_port()
    processes = [
        context.Process(
            target=_rank_main,
            args=(rank, port, args.sizes, tuple(args.dtypes), results),
        )
        for rank in range(WORLD_SIZE)
    ]
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
            records.append(results.get(timeout=5))
        except Empty:
            break
    records.sort(key=lambda item: int(item["rank"]))
    summary = {
        "transport": "HIP stream epochs + mapped peer-host fused add",
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
