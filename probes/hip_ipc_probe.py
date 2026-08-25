"""Test whether one Windows HIP process can map another GPU's allocation."""

from __future__ import annotations

import argparse
import ctypes
import json
import multiprocessing as mp
from pathlib import Path
from queue import Empty

from windows_amd_vllm_multigpu.hip_runtime import (
    HIP_MEMCPY_DEVICE_TO_HOST,
    HIP_MEMCPY_HOST_TO_DEVICE,
    HipRuntime,
)

SIZE_BYTES = 4096
TIMEOUT_SECONDS = 30
PATTERN = bytes((index * 17 + 31) & 0xFF for index in range(SIZE_BYTES))


def _exporter(
    connection: object,
    results: mp.Queue,
    runtime_path: str | None,
    dependency_dirs: list[str],
    export_device: int,
) -> None:
    pointer = 0
    record: dict[str, object] = {"rank": 0, "device": export_device}
    runtime = None
    try:
        runtime = HipRuntime(runtime_path, dependency_dirs)
        runtime.set_device(export_device)
        pointer = runtime.malloc(SIZE_BYTES)
        source = (ctypes.c_ubyte * SIZE_BYTES).from_buffer_copy(PATTERN)
        runtime.memcpy(
            pointer,
            ctypes.addressof(source),
            SIZE_BYTES,
            HIP_MEMCPY_HOST_TO_DEVICE,
        )
        handle = runtime.ipc_get_mem_handle(pointer)
        record["export_succeeded"] = True
        record["handle_size"] = len(handle)
        connection.send_bytes(handle)
        record["importer_reply"] = connection.recv()
    except Exception as error:  # noqa: BLE001 - preserve child-process diagnostics
        record["export_succeeded"] = False
        record["error"] = f"{type(error).__name__}: {error}"
        try:
            connection.send_bytes(b"")
        except Exception:  # noqa: BLE001, S110 - the peer may already be gone
            pass
    finally:
        if pointer and runtime is not None:
            try:
                runtime.free(pointer)
            except Exception as error:  # noqa: BLE001 - report cleanup failure
                record["free_error"] = f"{type(error).__name__}: {error}"
        connection.close()
        results.put(record)


def _importer(
    connection: object,
    results: mp.Queue,
    runtime_path: str | None,
    dependency_dirs: list[str],
    import_device: int,
) -> None:
    imported_pointer = 0
    record: dict[str, object] = {"rank": 1, "device": import_device}
    runtime = None
    try:
        handle = connection.recv_bytes()
        if not handle:
            raise RuntimeError("exporter could not create a HIP IPC handle")
        runtime = HipRuntime(runtime_path, dependency_dirs)
        runtime.set_device(import_device)
        imported_pointer = runtime.ipc_open_mem_handle(handle)
        record["import_succeeded"] = True
        destination = (ctypes.c_ubyte * SIZE_BYTES)()
        runtime.memcpy(
            ctypes.addressof(destination),
            imported_pointer,
            SIZE_BYTES,
            HIP_MEMCPY_DEVICE_TO_HOST,
        )
        record["data_correct"] = bytes(destination) == PATTERN
    except Exception as error:  # noqa: BLE001 - preserve child-process diagnostics
        record["import_succeeded"] = False
        record["data_correct"] = False
        record["error"] = f"{type(error).__name__}: {error}"
    finally:
        if imported_pointer and runtime is not None:
            try:
                runtime.ipc_close_mem_handle(imported_pointer)
            except Exception as error:  # noqa: BLE001 - report cleanup failure
                record["close_error"] = f"{type(error).__name__}: {error}"
        try:
            connection.send(record)
        except Exception:  # noqa: BLE001, S110 - the peer may already be gone
            pass
        connection.close()
        results.put(record)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dll", type=Path)
    parser.add_argument(
        "--dependency-dir",
        action="append",
        default=[],
        type=Path,
    )
    parser.add_argument("--export-device", type=int, default=0)
    parser.add_argument("--import-device", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    runtime_path = (
        str(arguments.runtime_dll.resolve()) if arguments.runtime_dll else None
    )
    dependency_dirs = [
        str(directory.resolve()) for directory in arguments.dependency_dir
    ]
    context = mp.get_context("spawn")
    exporter_connection, importer_connection = context.Pipe(duplex=True)
    results = context.Queue()
    processes = [
        context.Process(
            target=_exporter,
            args=(
                exporter_connection,
                results,
                runtime_path,
                dependency_dirs,
                arguments.export_device,
            ),
        ),
        context.Process(
            target=_importer,
            args=(
                importer_connection,
                results,
                runtime_path,
                dependency_dirs,
                arguments.import_device,
            ),
        ),
    ]
    for process in processes:
        process.start()
    exporter_connection.close()
    importer_connection.close()
    for process in processes:
        process.join(TIMEOUT_SECONDS)
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
        "runtime_dll": runtime_path,
        "export_device": arguments.export_device,
        "import_device": arguments.import_device,
        "size_bytes": SIZE_BYTES,
        "ranks": records,
        "exit_codes": [process.exitcode for process in processes],
    }
    summary["passed"] = (
        len(records) == 2
        and bool(records[0].get("export_succeeded"))
        and bool(records[1].get("import_succeeded"))
        and bool(records[1].get("data_correct"))
        and all(code == 0 for code in summary["exit_codes"])
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
