"""Probe Windows single-rank Gloo bootstrap without loading a GPU model."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import socket
import time
from datetime import timedelta

import torch
import torch.distributed as dist


def run_probe(
    host: str,
    interface: str,
    timeout: int | None,
    use_cuda: bool,
    use_accelerator: bool,
    use_vllm: bool,
) -> int:
    """Run one Gloo bootstrap in the current process."""
    if interface:
        os.environ["GLOO_SOCKET_IFNAME"] = interface
    else:
        os.environ.pop("GLOO_SOCKET_IFNAME", None)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((host, 0))
        port = listener.getsockname()[1]

    init_method = f"tcp://{host}:{port}"
    if use_cuda:
        if use_accelerator:
            torch.accelerator.set_device_index(torch.device("cuda:0"))
        else:
            torch.cuda.set_device(0)
        torch.empty(1, device="cuda")
    start = time.perf_counter()
    try:
        process_group_timeout = (
            None if timeout is None else timedelta(seconds=timeout)
        )
        if use_vllm:
            from vllm.distributed.parallel_state import (
                init_distributed_environment,
            )

            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=init_method,
                local_rank=0,
                backend="nccl",
                timeout=process_group_timeout,
            )
        else:
            dist.init_process_group(
                backend="gloo",
                init_method=init_method,
                world_size=1,
                rank=0,
                timeout=process_group_timeout,
            )
        elapsed = time.perf_counter() - start
        print(
            "PASS "
            f"backend={dist.get_backend()} host={host} "
            f"interface={interface or '<auto>'} cuda={use_cuda} "
            f"accelerator={use_accelerator} vllm={use_vllm} "
            f"elapsed={elapsed:.3f}s"
        )
        return 0
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def spawned_probe(
    host: str,
    interface: str,
    timeout: int | None,
    use_cuda: bool,
    use_accelerator: bool,
    use_vllm: bool,
    result_queue: mp.Queue,
) -> None:
    """Run the probe in a fresh Windows spawned child."""
    try:
        result_queue.put(
            run_probe(
                host,
                interface,
                timeout,
                use_cuda,
                use_accelerator,
                use_vllm,
            )
        )
    except BaseException as exc:
        result_queue.put(repr(exc))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--interface", default="")
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--no-timeout", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--parent-cuda", action="store_true")
    parser.add_argument("--accelerator", action="store_true")
    parser.add_argument("--vllm", action="store_true")
    parser.add_argument("--spawn", action="store_true")
    args = parser.parse_args()

    if not args.spawn:
        return run_probe(
            args.host,
            args.interface,
            None if args.no_timeout else args.timeout,
            args.cuda,
            args.accelerator,
            args.vllm,
        )

    if args.parent_cuda:
        torch.cuda.set_device(0)
        torch.empty(1, device="cuda")

    context = mp.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=spawned_probe,
        args=(
            args.host,
            args.interface,
            None if args.no_timeout else args.timeout,
            args.cuda,
            args.accelerator,
            args.vllm,
            result_queue,
        ),
    )
    process.start()
    process.join(args.timeout + 10)
    if process.is_alive():
        process.terminate()
        process.join(5)
        raise TimeoutError("spawned Gloo probe did not exit")
    if result_queue.empty():
        raise RuntimeError(f"spawned probe exited with code {process.exitcode}")
    result = result_queue.get()
    if result != 0:
        raise RuntimeError(f"spawned probe failed: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
