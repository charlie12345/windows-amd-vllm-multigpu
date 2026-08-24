"""Measure single-process cross-GPU and pinned-host transport ceilings."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable

import torch


DEFAULT_SIZES = [4 * 1024, 64 * 1024, 1024 * 1024, 8 * 1024 * 1024]


def _iterations(size_bytes: int) -> int:
    if size_bytes <= 64 * 1024:
        return 100
    if size_bytes <= 1024 * 1024:
        return 30
    return 10


def _sync_both() -> None:
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)


def _measure(operation: Callable[[], None], iterations: int) -> float:
    for _ in range(3):
        operation()
    _sync_both()
    started = time.perf_counter()
    for _ in range(iterations):
        operation()
    _sync_both()
    return (time.perf_counter() - started) * 1000.0 / iterations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="*", type=int, default=DEFAULT_SIZES)
    args = parser.parse_args()

    if torch.cuda.device_count() < 2:
        raise SystemExit("two GPUs are required")

    records = []
    for size_bytes in args.sizes:
        count = max(1, size_bytes // 4)
        iterations = _iterations(size_bytes)
        source0 = torch.full((count,), 1.0, dtype=torch.float32, device="cuda:0")
        source1 = torch.full((count,), 2.0, dtype=torch.float32, device="cuda:1")
        destination0 = torch.empty_like(source0)
        destination1 = torch.empty_like(source1)
        stage0 = torch.empty((count,), dtype=torch.float32, pin_memory=True)
        stage1 = torch.empty((count,), dtype=torch.float32, pin_memory=True)
        stage_sum = torch.empty((count,), dtype=torch.float32, pin_memory=True)

        def cross_device_copy() -> None:
            destination1.copy_(source0, non_blocking=False)

        def explicit_host_copy() -> None:
            stage0.copy_(source0, non_blocking=False)
            destination1.copy_(stage0, non_blocking=False)

        def cross_device_all_reduce() -> None:
            destination0.copy_(source1, non_blocking=False)
            destination0.add_(source0)
            destination1.copy_(destination0, non_blocking=False)

        def pinned_host_all_reduce() -> None:
            stage0.copy_(source0, non_blocking=False)
            stage1.copy_(source1, non_blocking=False)
            torch.add(stage0, stage1, out=stage_sum)
            destination0.copy_(stage_sum, non_blocking=False)
            destination1.copy_(stage_sum, non_blocking=False)

        timings = {
            "driver_cross_device_copy_ms": _measure(cross_device_copy, iterations),
            "explicit_host_copy_ms": _measure(explicit_host_copy, iterations),
            "driver_cross_device_all_reduce_ms": _measure(
                cross_device_all_reduce, iterations
            ),
            "pinned_host_all_reduce_ms": _measure(pinned_host_all_reduce, iterations),
        }
        cross_device_copy()
        _sync_both()
        correctness = {
            "driver_cross_device_copy": bool(torch.all(destination1 == 1.0).item()),
        }
        cross_device_all_reduce()
        _sync_both()
        correctness["driver_cross_device_all_reduce"] = bool(
            torch.all(destination0 == 3.0).item()
            and torch.all(destination1 == 3.0).item()
        )
        pinned_host_all_reduce()
        _sync_both()
        correctness["pinned_host_all_reduce"] = bool(
            torch.all(destination0 == 3.0).item()
            and torch.all(destination1 == 3.0).item()
        )
        records.append(
            {
                "size_bytes": size_bytes,
                "iterations": iterations,
                "timings": timings,
                "correctness": correctness,
            }
        )

    summary = {
        "peer_access_0_to_1": torch.cuda.can_device_access_peer(0, 1),
        "peer_access_1_to_0": torch.cuda.can_device_access_peer(1, 0),
        "records": records,
    }
    summary["passed"] = all(
        all(record["correctness"].values()) for record in records
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
