"""Exercise native Windows RCCL collectives with two processes/two AMD GPUs."""

from __future__ import annotations

import ctypes
import json
import multiprocessing as mp
import os
import struct
import time
from pathlib import Path
from queue import Empty


COUNT = int(os.environ.get("WAVMG_RCCL_COUNT", str(1024 * 1024)))
ITERATIONS = int(os.environ.get("WAVMG_RCCL_ITERATIONS", "1"))
if ITERATIONS < 1:
    raise ValueError("WAVMG_RCCL_ITERATIONS must be at least 1")
WARMUP_ITERATIONS = int(os.environ.get("WAVMG_RCCL_WARMUP_ITERATIONS", "0"))
if WARMUP_ITERATIONS < 0:
    raise ValueError("WAVMG_RCCL_WARMUP_ITERATIONS cannot be negative")
TIMEOUT_SECONDS = 180
WORLD_SIZE = int(os.environ.get("WAVMG_RCCL_WORLD_SIZE", "2"))
OPERATION = os.environ.get("WAVMG_RCCL_OPERATION", "all_reduce").lower()
if OPERATION not in {"all_reduce", "all_gather", "reduce_scatter", "broadcast"}:
    raise ValueError(
        "Unsupported WAVMG_RCCL_OPERATION; choose all_reduce, all_gather, "
        "reduce_scatter, or broadcast"
    )
DType = str
DTYPE = os.environ.get("WAVMG_RCCL_DTYPE", "f32").lower()
DTYPE_CONFIG: dict[DType, tuple[int, int]] = {
    "f16": (6, 2),
    "f32": (7, 4),
    "bf16": (9, 2),
}
if DTYPE not in DTYPE_CONFIG:
    raise ValueError(
        f"Unsupported WAVMG_RCCL_DTYPE={DTYPE!r}; choose f16, f32, or bf16"
    )
NCCL_DATA_TYPE, ELEMENT_SIZE = DTYPE_CONFIG[DTYPE]
NCCL_SUM = 0
HIP_MEMCPY_HOST_TO_DEVICE = 1
HIP_MEMCPY_DEVICE_TO_HOST = 2
NCCL_ALL_REDUCE_IMPL_SYMBOL = (
    "?ncclAllReduce_impl@@YA?AW4ncclResult_t@@PEBXPEAX_KW4ncclDataType_t@@"
    "W4ncclRedOp_t@@PEAUncclComm@@PEAUihipStream_t@@@Z"
)


class NcclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_char * 128)]


def _encode_scalar(value: float) -> bytes:
    if DTYPE == "f32":
        return struct.pack("<f", value)
    if DTYPE == "f16":
        return struct.pack("<e", value)
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    # IEEE round-to-nearest-even while truncating float32 to bfloat16.
    bits += 0x7FFF + ((bits >> 16) & 1)
    return struct.pack("<H", (bits >> 16) & 0xFFFF)


def _decode_scalar(raw: bytes) -> float:
    if DTYPE == "f32":
        return float(struct.unpack("<f", raw)[0])
    if DTYPE == "f16":
        return float(struct.unpack("<e", raw)[0])
    bits = struct.unpack("<H", raw)[0] << 16
    return float(struct.unpack("<f", struct.pack("<I", bits))[0])


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_rccl() -> ctypes.CDLL:
    root = _project_root()
    rocm_bin = root / ".venv" / "Lib" / "site-packages" / "_rocm_sdk_devel" / "bin"
    torch_lib = root / ".venv" / "Lib" / "site-packages" / "torch" / "lib"
    for directory in (rocm_bin, torch_lib):
        if directory.is_dir():
            os.add_dll_directory(str(directory))

    library = ctypes.CDLL(str(root / "build" / "rccl-windows" / "rccl.dll"))
    library.ncclGetUniqueId.argtypes = [ctypes.POINTER(NcclUniqueId)]
    library.ncclGetUniqueId.restype = ctypes.c_int
    library.ncclCommInitRank.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
        NcclUniqueId,
        ctypes.c_int,
    ]
    library.ncclCommInitRank.restype = ctypes.c_int
    all_reduce = (
        getattr(library, NCCL_ALL_REDUCE_IMPL_SYMBOL)
        if os.environ.get("WAVMG_RCCL_DIRECT_IMPL") == "1"
        else library.ncclAllReduce
    )
    all_reduce.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    all_reduce.restype = ctypes.c_int
    library.wavmg_all_reduce = all_reduce
    library.ncclAllGather.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    library.ncclAllGather.restype = ctypes.c_int
    library.ncclReduceScatter.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    library.ncclReduceScatter.restype = ctypes.c_int
    library.ncclBroadcast.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    library.ncclBroadcast.restype = ctypes.c_int
    library.ncclCommDestroy.argtypes = [ctypes.c_void_p]
    library.ncclCommDestroy.restype = ctypes.c_int
    library.ncclCommAbort.argtypes = [ctypes.c_void_p]
    library.ncclCommAbort.restype = ctypes.c_int
    library.ncclGetErrorString.argtypes = [ctypes.c_int]
    library.ncclGetErrorString.restype = ctypes.c_char_p
    return library


def _check(library: ctypes.CDLL, result: int, operation: str) -> None:
    if result == 0:
        return
    raw = library.ncclGetErrorString(result)
    message = raw.decode("utf-8", errors="replace") if raw else "unknown RCCL error"
    raise RuntimeError(f"{operation} failed: {result} ({message})")


def _launch_collective(
    library: ctypes.CDLL, send_pointer: int, receive_pointer: int, comm: ctypes.c_void_p
) -> tuple[int, str]:
    send = ctypes.c_void_p(send_pointer)
    receive = ctypes.c_void_p(receive_pointer)
    stream = ctypes.c_void_p()
    if OPERATION == "all_reduce":
        return (
            library.wavmg_all_reduce(
                send, receive, COUNT, NCCL_DATA_TYPE, NCCL_SUM, comm, stream
            ),
            "ncclAllReduce",
        )
    if OPERATION == "all_gather":
        return (
            library.ncclAllGather(
                send, receive, COUNT, NCCL_DATA_TYPE, comm, stream
            ),
            "ncclAllGather",
        )
    if OPERATION == "reduce_scatter":
        return (
            library.ncclReduceScatter(
                send, receive, COUNT, NCCL_DATA_TYPE, NCCL_SUM, comm, stream
            ),
            "ncclReduceScatter",
        )
    return (
        library.ncclBroadcast(
            send, receive, COUNT, NCCL_DATA_TYPE, 0, comm, stream
        ),
        "ncclBroadcast",
    )


def _rank_main(rank: int, unique_id_bytes: bytes, results: mp.Queue) -> None:
    device = int(os.environ.get("WAVMG_RCCL_DEVICE", str(rank)))
    record: dict[str, object] = {"rank": rank, "device": device}
    library = None
    comm = ctypes.c_void_p()
    init_returned = False
    collective_called = False
    successful = False
    rank_log = None
    hip = None
    send_pointer = 0
    receive_pointer = 0
    try:
        rank_log_dir = os.environ.get("WAVMG_RCCL_RANK_LOG_DIR")
        if rank_log_dir:
            log_dir = Path(rank_log_dir).resolve()
            log_dir.mkdir(parents=True, exist_ok=True)
            rank_log = (log_dir / f"rank-{rank}.log").open(
                "w", encoding="utf-8", buffering=1
            )
            os.dup2(rank_log.fileno(), 1)
            os.dup2(rank_log.fileno(), 2)

        os.environ.setdefault("NCCL_DEBUG", "INFO")
        os.environ.setdefault("NCCL_RAS_ENABLE", "0")

        import torch
        from windows_amd_vllm_multigpu.hip_runtime import HipRuntime

        torch.cuda.set_device(device)
        hip = HipRuntime()
        hip.set_device(device)
        library = _load_rccl()
        unique_id = NcclUniqueId.from_buffer_copy(unique_id_bytes)
        _check(
            library,
            library.ncclCommInitRank(
                ctypes.byref(comm), WORLD_SIZE, unique_id, rank
            ),
            "ncclCommInitRank",
        )
        init_returned = True
        pre_all_reduce_delay = float(
            os.environ.get("WAVMG_RCCL_PRE_ALLREDUCE_DELAY", "0")
        )
        if pre_all_reduce_delay > 0:
            time.sleep(pre_all_reduce_delay)

        value = float(rank + 1)
        send_elements = COUNT * (WORLD_SIZE if OPERATION == "reduce_scatter" else 1)
        receive_elements = COUNT * (WORLD_SIZE if OPERATION == "all_gather" else 1)
        send_byte_count = send_elements * ELEMENT_SIZE
        receive_byte_count = receive_elements * ELEMENT_SIZE
        send_host = ctypes.create_string_buffer(
            _encode_scalar(value) * send_elements
        )
        receive_host = ctypes.create_string_buffer(receive_byte_count)
        send_pointer = hip.malloc(send_byte_count)
        receive_pointer = hip.malloc(receive_byte_count)
        hip.memcpy(
            send_pointer,
            ctypes.addressof(send_host),
            send_byte_count,
            HIP_MEMCPY_HOST_TO_DEVICE,
        )
        collective_called = True
        for _ in range(WARMUP_ITERATIONS):
            result, operation_name = _launch_collective(
                library, send_pointer, receive_pointer, comm
            )
            _check(library, result, operation_name)
        hip.device_synchronize()
        started = time.perf_counter()
        for _ in range(ITERATIONS):
            result, operation_name = _launch_collective(
                library, send_pointer, receive_pointer, comm
            )
            _check(library, result, operation_name)
        hip.device_synchronize()
        elapsed_seconds = time.perf_counter() - started
        hip.memcpy(
            ctypes.addressof(receive_host),
            receive_pointer,
            receive_byte_count,
            HIP_MEMCPY_DEVICE_TO_HOST,
        )

        expected = float(WORLD_SIZE * (WORLD_SIZE + 1) // 2)
        received = receive_host.raw[:receive_byte_count]
        if OPERATION == "all_gather":
            expected_payload = b"".join(
                _encode_scalar(float(peer + 1)) * COUNT
                for peer in range(WORLD_SIZE)
            )
        elif OPERATION == "broadcast":
            expected = 1.0
            expected_payload = _encode_scalar(expected) * COUNT
        else:
            expected_payload = _encode_scalar(expected) * receive_elements
        correct = received == expected_payload
        first_value = _decode_scalar(received[:ELEMENT_SIZE])
        maximum_error = 0.0 if correct else abs(first_value - expected)
        record.update(
            {
                "initialized": True,
                "comm_allocated": True,
                "collective_returned": True,
                "correct": correct,
                "maximum_error": maximum_error,
                "first_value": first_value,
                "iterations": ITERATIONS,
                "warmup_iterations": WARMUP_ITERATIONS,
                "elapsed_seconds": elapsed_seconds,
                "latency_ms": elapsed_seconds * 1000.0 / ITERATIONS,
                "comm": int(comm.value or 0),
            }
        )
        successful = bool(record["correct"])
    except Exception as error:
        record.update(
            {
                "initialized": init_returned,
                "comm_allocated": bool(comm.value),
                "collective_called": collective_called,
                "collective_returned": False,
                "correct": False,
                "error": f"{type(error).__name__}: {error}",
            }
        )
    finally:
        if hip is not None:
            for pointer in (receive_pointer, send_pointer):
                if pointer:
                    try:
                        hip.free(pointer)
                    except Exception as error:
                        record.setdefault(
                            "memory_cleanup_errors", []
                        ).append(f"{type(error).__name__}: {error}")
        if library is not None and comm.value:
            try:
                if successful:
                    result = library.ncclCommDestroy(comm)
                    record["destroy_result"] = int(result)
                else:
                    result = library.ncclCommAbort(comm)
                    record["abort_result"] = int(result)
            except Exception as error:
                record["cleanup_error"] = f"{type(error).__name__}: {error}"
        results.put(record)
        if rank_log is not None:
            rank_log.flush()
            rank_log.close()


def main() -> int:
    os.environ.setdefault("NCCL_DEBUG", "INFO")
    os.environ.setdefault("NCCL_RAS_ENABLE", "0")
    library = _load_rccl()
    unique_id = NcclUniqueId()
    _check(library, library.ncclGetUniqueId(ctypes.byref(unique_id)), "ncclGetUniqueId")
    unique_id_bytes = bytes(unique_id)

    context = mp.get_context("spawn")
    results = context.Queue()
    processes = [
        context.Process(target=_rank_main, args=(rank, unique_id_bytes, results))
        for rank in range(WORLD_SIZE)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(TIMEOUT_SECONDS)
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
        "operation": OPERATION,
        "dtype": DTYPE,
        "count": COUNT,
        "iterations": ITERATIONS,
        "warmup_iterations": WARMUP_ITERATIONS,
        "send_bytes_per_rank": COUNT
        * (WORLD_SIZE if OPERATION == "reduce_scatter" else 1)
        * ELEMENT_SIZE,
        "receive_bytes_per_rank": COUNT
        * (WORLD_SIZE if OPERATION == "all_gather" else 1)
        * ELEMENT_SIZE,
        "world_size": WORLD_SIZE,
        "ranks": records,
        "exit_codes": [process.exitcode for process in processes],
    }
    summary["passed"] = (
        len(records) == WORLD_SIZE
        and all(bool(record.get("initialized")) for record in records)
        and all(bool(record.get("collective_returned")) for record in records)
        and all(bool(record.get("correct")) for record in records)
        and all(code == 0 for code in summary["exit_codes"])
    )
    rendered = json.dumps(summary, indent=2)
    print(rendered)
    result_file = os.environ.get("WAVMG_RCCL_RESULT_FILE")
    if result_file:
        destination = Path(result_file).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n", encoding="utf-8")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
