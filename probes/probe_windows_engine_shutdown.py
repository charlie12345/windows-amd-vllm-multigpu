"""Validate the spawn-safe cooperative EngineCore shutdown primitive."""

from __future__ import annotations

import multiprocessing
import sys


def _cooperative_child(shutdown_event) -> None:
    if not shutdown_event.wait(timeout=10):
        raise TimeoutError("parent did not request cooperative shutdown")


def main() -> int:
    if sys.platform != "win32":
        print("SKIP: Windows-only cooperative shutdown probe")
        return 0

    from vllm.v1.engine.utils import _shutdown_core_processes

    context = multiprocessing.get_context("spawn")
    shutdown_event = context.Event()
    process = context.Process(
        target=_cooperative_child,
        args=(shutdown_event,),
        name="WavmgCooperativeShutdownProbe",
    )
    process.start()
    _shutdown_core_processes([process], shutdown_event, timeout=5)
    process.join(timeout=1)

    if process.is_alive() or process.exitcode != 0:
        raise RuntimeError(
            "cooperative shutdown failed: "
            f"alive={process.is_alive()} exitcode={process.exitcode}"
        )

    print("PASS: Windows spawn child exited through cooperative shutdown")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
