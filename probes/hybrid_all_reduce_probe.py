"""Validate D3D12 fast-path selection and RCCL overflow routing."""

from __future__ import annotations

from datetime import timedelta
import json
import multiprocessing as mp
from queue import Empty
import socket


WORLD_SIZE = 2
TIMEOUT_SECONDS = 180
D3D12_LIMIT = 64 * 1024 * 1024
SMALL_BYTES = 1024 * 1024
OVERFLOW_BYTES = 80 * 1024 * 1024


def _unused_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _rank_main(rank: int, port: int, results: mp.Queue) -> None:
    import torch
    import torch.distributed as dist

    from windows_amd_vllm_multigpu.d3d12_all_reduce import D3D12AllReduce
    from windows_amd_vllm_multigpu.rccl import RcclCommunicator

    record: dict[str, object] = {"rank": rank, "device": rank}
    d3d12 = None
    rccl = None
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
        d3d12 = D3D12AllReduce(
            group=None, device=rank, max_size_bytes=D3D12_LIMIT
        )
        rccl = RcclCommunicator(group=None, device=rank)
        stream = torch.cuda.Stream(device=rank)
        cases = []
        for size_bytes in (SMALL_BYTES, OVERFLOW_BYTES):
            source = torch.full(
                (size_bytes // 2,),
                float(rank + 1),
                dtype=torch.float16,
                device=f"cuda:{rank}",
            )
            torch.cuda.synchronize(rank)
            use_d3d12 = d3d12.can_handle(source)
            with torch.cuda.stream(stream):
                output = (
                    d3d12.all_reduce(source)
                    if use_d3d12
                    else rccl.all_reduce(source)
                )
            stream.synchronize()
            correct = bool(torch.all(output.float() == 3.0).item())
            cases.append(
                {
                    "size_bytes": size_bytes,
                    "backend": "d3d12" if use_d3d12 else "rccl-overflow",
                    "correct": correct,
                }
            )
        record["cases"] = cases
        record["passed"] = (
            cases[0]["backend"] == "d3d12"
            and cases[1]["backend"] == "rccl-overflow"
            and all(bool(case["correct"]) for case in cases)
        )
    except Exception as error:
        record["passed"] = False
        record["error"] = f"{type(error).__name__}: {error}"
    finally:
        if d3d12 is not None:
            try:
                d3d12.destroy()
            except Exception as error:
                record["d3d12_destroy_error"] = f"{type(error).__name__}: {error}"
                record["passed"] = False
        if rccl is not None:
            try:
                rccl.destroy()
            except Exception as error:
                record["rccl_destroy_error"] = f"{type(error).__name__}: {error}"
                record["passed"] = False
        if "dist" in locals() and dist.is_initialized():
            dist.destroy_process_group()
        results.put(record)


def main() -> int:
    context = mp.get_context("spawn")
    results = context.Queue()
    port = _unused_local_port()
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
        "world_size": WORLD_SIZE,
        "d3d12_limit_bytes": D3D12_LIMIT,
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
