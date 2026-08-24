"""Prove two-rank Gloo and a correctness-first host-staged GPU all-reduce."""

from __future__ import annotations

import json
import multiprocessing as mp
import socket
from datetime import timedelta
from queue import Empty


WORLD_SIZE = 2
TIMEOUT_SECONDS = 45


def _unused_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _rank_main(rank: int, port: int, results: mp.Queue) -> None:
    import torch
    import torch.distributed as dist

    from windows_amd_vllm_multigpu import HostStagedGloo

    record: dict[str, object] = {"rank": rank}
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=WORLD_SIZE,
            timeout=timedelta(seconds=TIMEOUT_SECONDS),
        )

        cpu_value = torch.tensor([float(rank + 1)], dtype=torch.float32)
        dist.all_reduce(cpu_value, op=dist.ReduceOp.SUM)
        record["cpu_all_reduce"] = cpu_value.item()

        communicator = HostStagedGloo(device=rank)
        gpu_value = torch.tensor(
            [float(rank + 1)], dtype=torch.float32, device=f"cuda:{rank}"
        )
        reduced = communicator.all_reduce(gpu_value)
        record["host_staged_gpu_all_reduce"] = reduced.cpu().item()

        gather_input = torch.tensor(
            [float(rank * 2 + 1), float(rank * 2 + 2)], device=f"cuda:{rank}"
        )
        gathered = communicator.all_gather(gather_input, dim=0)
        record["host_staged_gpu_all_gather"] = gathered.cpu().tolist()

        scatter_input = torch.full((4,), float(rank + 1), device=f"cuda:{rank}")
        scattered = communicator.reduce_scatter(scatter_input, dim=0)
        record["host_staged_gpu_reduce_scatter"] = scattered.cpu().tolist()

        broadcast_value = torch.tensor(
            [7.0 if rank == 0 else -1.0], device=f"cuda:{rank}"
        )
        communicator.broadcast(broadcast_value, src=0)
        record["host_staged_gpu_broadcast"] = broadcast_value.cpu().item()

        if rank == 0:
            communicator.send(torch.tensor([11.0], device="cuda:0"), dst=1)
            point_to_point = 11.0
        else:
            point_to_point = communicator.recv((1,), torch.float32, src=0).cpu().item()
        communicator.barrier()
        record["host_staged_gpu_point_to_point"] = point_to_point

        # Do not pass a HIP tensor directly to Gloo on Windows. The current
        # PyTorch wheel terminates the process with native exception 0xC0000005
        # instead of raising a catchable Python exception. Keep this probe safe
        # and exercise only the explicit CPU-staging path above.
        record["direct_gpu_gloo"] = {
            "attempted": False,
            "reason": "Known fatal ProcessGroupGloo GPU-tensor registry failure",
        }

        expected = float(sum(range(1, WORLD_SIZE + 1)))
        record["passed"] = (
            record["cpu_all_reduce"] == expected
            and record["host_staged_gpu_all_reduce"] == expected
            and record["host_staged_gpu_all_gather"] == [1.0, 2.0, 3.0, 4.0]
            and record["host_staged_gpu_reduce_scatter"] == [3.0, 3.0]
            and record["host_staged_gpu_broadcast"] == 7.0
            and record["host_staged_gpu_point_to_point"] == 11.0
        )
    except Exception as error:
        record["passed"] = False
        record["fatal_error"] = f"{type(error).__name__}: {error}"
    finally:
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
