"""Uncached reads and chunked writes for the Flash-Next pack builders.

`Reader` reads a pack uncached (`F_NOCACHE` `pread`), as the engine does, into 16 KiB page-aligned
buffers; `ShardReader` reads the checkpoint shards a pack is verified against. The format itself
is checkpoint/pack_format.py's.
"""

from __future__ import annotations

import fcntl
import os

import numpy as np

from jitengine_qwen3_8_flash_next_m2.checkpoint.pack_format import PAGE, PackError, align_up
from jitengine_qwen3_8_flash_next_m2.moe.expert_store import F_NOCACHE


def aligned_buf(nbytes: int, align: int = PAGE) -> np.ndarray:
    raw = np.empty(nbytes + align, dtype=np.uint8)
    off = (-raw.ctypes.data) % align
    return raw[off : off + nbytes]


class Reader:
    """`pread` from a pack with the page cache switched off, the way the engine's pool fill reads."""

    def __init__(self, path: str, nocache: bool = True):
        self.path = path
        self.fd = os.open(path, os.O_RDONLY)
        if nocache:
            fcntl.fcntl(self.fd, F_NOCACHE, 1)

    def read(self, offset: int, length: int) -> bytes:
        buf = aligned_buf(align_up(length))[:length]
        mv = memoryview(buf).cast("B")
        got = 0
        while got < length:
            n = os.preadv(self.fd, [mv[got:]], offset + got)
            if n <= 0:
                raise PackError(
                    f"{self.path}: short read at {offset + got} (+{length - got} wanted)"
                )
            got += n
        return bytes(buf)

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


class ShardReader:
    """Cached read-only fds over the checkpoint shards, for byte-for-byte pack verification."""

    def __init__(self):
        self.fds: dict[str, int] = {}

    def read(self, path: str, offset: int, length: int) -> bytes:
        fd = self.fds.get(path)
        if fd is None:
            fd = self.fds[path] = os.open(path, os.O_RDONLY)
        buf = bytearray(length)
        mv = memoryview(buf)
        got = 0
        while got < length:
            n = os.preadv(fd, [mv[got:]], offset + got)
            if n <= 0:
                raise PackError(f"{path}: short read at {offset + got}")
            got += n
        return bytes(buf)

    def close(self) -> None:
        for fd in self.fds.values():
            os.close(fd)
        self.fds.clear()


def write_all(fd: int, mv: memoryview, chunk_bytes: int = 32 << 20) -> None:
    written = 0
    while written < len(mv):
        written += os.write(fd, mv[written : written + chunk_bytes])
