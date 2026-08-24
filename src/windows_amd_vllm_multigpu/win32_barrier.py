"""Low-overhead two-process barrier built from Windows auto-reset events."""

from __future__ import annotations

import ctypes


WAIT_OBJECT_0 = 0
WAIT_TIMEOUT = 258
INFINITE = 0xFFFFFFFF


class Win32PairBarrier:
    """A reusable barrier for exactly two ranks.

    Each rank owns one named auto-reset event and waits on the peer's event.
    There is exactly one waiter per event, so a successful wait consumes the
    signal and makes the next generation safe without an explicit reset.
    """

    def __init__(self, name_prefix: str, rank: int, timeout_ms: int = 30_000) -> None:
        if rank not in (0, 1):
            raise ValueError("Win32PairBarrier supports ranks 0 and 1 only")
        self.rank = rank
        self.timeout_ms = timeout_ms
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._kernel32.CreateEventW.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_wchar_p,
        ]
        self._kernel32.CreateEventW.restype = ctypes.c_void_p
        self._kernel32.SetEvent.argtypes = [ctypes.c_void_p]
        self._kernel32.SetEvent.restype = ctypes.c_int
        self._kernel32.WaitForSingleObject.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
        ]
        self._kernel32.WaitForSingleObject.restype = ctypes.c_uint
        self._kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        self._kernel32.CloseHandle.restype = ctypes.c_int

        self._handles = []
        for event_rank in range(2):
            name = f"Local\\{name_prefix}_rank{event_rank}"
            handle = self._kernel32.CreateEventW(None, False, False, name)
            if not handle:
                raise ctypes.WinError(ctypes.get_last_error())
            self._handles.append(handle)

    def wait(self) -> None:
        own = self._handles[self.rank]
        peer = self._handles[1 - self.rank]
        if not self._kernel32.SetEvent(own):
            raise ctypes.WinError(ctypes.get_last_error())
        result = self._kernel32.WaitForSingleObject(peer, self.timeout_ms)
        if result == WAIT_TIMEOUT:
            raise TimeoutError("the peer did not reach the Windows event barrier")
        if result != WAIT_OBJECT_0:
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self) -> None:
        while self._handles:
            handle = self._handles.pop()
            self._kernel32.CloseHandle(handle)

    def __enter__(self) -> "Win32PairBarrier":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
