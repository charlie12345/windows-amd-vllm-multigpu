"""Validate the Windows RCCL wrapper on real PyTorch HIP tensors/streams."""

from __future__ import annotations

import json
import multiprocessing as mp
import socket
from datetime import timedelta
from queue import Empty


WORLD_SIZE = 2
TIMEOUT_SECONDS = 90


def _unused_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _rank_main(rank: int, port: int, results: mp.Queue) -> None:
    import torch
    import torch.distributed as dist

    from windows_amd_vllm_multigpu.rccl import RcclCommunicator

    record: dict[str, object] = {"rank": rank, "device": rank}
    communicator = None
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=WORLD_SIZE,
            timeout=timedelta(seconds=TIMEOUT_SECONDS),
        )
        communicator = RcclCommunicator(group=None, device=rank)
        stream = torch.cuda.Stream(device=rank)
        dtype_results: dict[str, object] = {}
        for dtype in (torch.float16, torch.float32, torch.bfloat16):
            with torch.cuda.stream(stream):
                value = torch.full(
                    (4096,), float(rank + 1), dtype=dtype, device=f"cuda:{rank}"
                )
                reduced = communicator.all_reduce(value)

                gather_input = (
                    torch.arange(1, 5, dtype=torch.float32, device=f"cuda:{rank}")
                    .reshape(2, 2)
                    .add_(rank * 4)
                    .to(dtype)
                )
                gathered = communicator.all_gather(gather_input, dim=-1)

                scatter_input = torch.full(
                    (2, 4), float(rank + 1), dtype=dtype, device=f"cuda:{rank}"
                )
                scattered = communicator.reduce_scatter(scatter_input, dim=-1)

                broadcast_input = torch.tensor(
                    [[7.0, 8.0]] if rank == 0 else [[-1.0, -1.0]],
                    dtype=dtype,
                    device=f"cuda:{rank}",
                )
                broadcast = communicator.broadcast(broadcast_input, src=0)
            stream.synchronize()

            reduced_ok = bool(torch.all(reduced == 3).item())
            gathered_values = gathered.float().cpu().tolist()
            scattered_values = scattered.float().cpu().tolist()
            broadcast_values = broadcast.float().cpu().tolist()
            dtype_results[str(dtype)] = {
                "all_reduce": reduced_ok,
                "all_gather": gathered_values,
                "reduce_scatter": scattered_values,
                "broadcast": broadcast_values,
                "passed": reduced_ok
                and gathered_values == [[1.0, 2.0, 5.0, 6.0], [3.0, 4.0, 7.0, 8.0]]
                and scattered_values == [[3.0, 3.0], [3.0, 3.0]]
                and broadcast_values == [[7.0, 8.0]],
            }
        record["dtypes"] = dtype_results
        record["passed"] = all(
            bool(value["passed"]) for value in dtype_results.values()
        )
    except Exception as error:
        record["passed"] = False
        record["fatal_error"] = f"{type(error).__name__}: {error}"
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
    results = context.Queue()
    port = _unused_local_port()
    processes = [
        context.Process(target=_rank_main, args=(rank, port, results))
        for rank in range(WORLD_SIZE)
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
