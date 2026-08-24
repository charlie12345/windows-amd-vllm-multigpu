"""Measure two-rank host-staged GPU all-reduce latency and logical bandwidth."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import socket
import time
from datetime import timedelta
from queue import Empty


WORLD_SIZE = 2
TIMEOUT_SECONDS = 120
DEFAULT_SIZES = [4 * 1024, 64 * 1024, 1024 * 1024, 8 * 1024 * 1024]


def _unused_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _iterations(size_bytes: int) -> int:
    if size_bytes <= 64 * 1024:
        return 50
    if size_bytes <= 1024 * 1024:
        return 20
    return 8


def _rank_main(rank: int, port: int, sizes: list[int], results: mp.Queue) -> None:
    import torch
    import torch.distributed as dist

    from windows_amd_vllm_multigpu import HostStagedGloo

    rank_results: list[dict[str, object]] = []
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group(
            "gloo",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=WORLD_SIZE,
            timeout=timedelta(seconds=TIMEOUT_SECONDS),
        )
        communicator = HostStagedGloo(device=rank)

        for size_bytes in sizes:
            element_count = max(1, size_bytes // 4)
            source = torch.full(
                (element_count,), float(rank + 1), dtype=torch.float32, device=rank
            )
            expected = float(sum(range(1, WORLD_SIZE + 1)))
            iterations = _iterations(size_bytes)

            for _ in range(3):
                output = communicator.all_reduce(source)
            torch.cuda.synchronize(rank)
            communicator.barrier()

            started = time.perf_counter()
            for _ in range(iterations):
                output = communicator.all_reduce(source)
            torch.cuda.synchronize(rank)
            communicator.barrier()
            elapsed = time.perf_counter() - started

            latency_ms = elapsed * 1000.0 / iterations
            logical_mib_s = (size_bytes / (1024 * 1024)) / (latency_ms / 1000.0)
            correct = bool(torch.all(output == expected).item())
            rank_results.append(
                {
                    "size_bytes": size_bytes,
                    "iterations": iterations,
                    "latency_ms": latency_ms,
                    "logical_mib_s": logical_mib_s,
                    "correct": correct,
                }
            )
        results.put({"rank": rank, "passed": True, "results": rank_results})
    except Exception as error:
        results.put(
            {
                "rank": rank,
                "passed": False,
                "error": f"{type(error).__name__}: {error}",
            }
        )
    finally:
        if "dist" in locals() and dist.is_initialized():
            dist.destroy_process_group()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="*", type=int, default=DEFAULT_SIZES)
    args = parser.parse_args()

    context = mp.get_context("spawn")
    queue = context.Queue()
    port = _unused_local_port()
    processes = [
        context.Process(target=_rank_main, args=(rank, port, args.sizes, queue))
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
            records.append(queue.get(timeout=5))
        except Empty:
            break
    records.sort(key=lambda item: int(item["rank"]))

    summary = {
        "transport": "pinned-host-stage + Gloo",
        "world_size": WORLD_SIZE,
        "ranks": records,
        "exit_codes": [process.exitcode for process in processes],
    }
    summary["passed"] = (
        len(records) == WORLD_SIZE
        and all(bool(record.get("passed")) for record in records)
        and all(
            bool(item["correct"])
            for record in records
            for item in record.get("results", [])
        )
        and all(code == 0 for code in summary["exit_codes"])
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())

