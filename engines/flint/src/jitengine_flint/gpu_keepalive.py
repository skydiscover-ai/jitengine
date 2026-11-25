"""GPU clock keep-alive for offload decode: a child process that streams tiny kernels while a job runs.

Offload decode submits work in bursts (one eval per MoE layer, host reads and bookkeeping between them). Under that
pattern the M2's GPU DVFS holds ~800 of 1398 MHz at ~90% residency, and the CPU clusters idle low too; a continuous
stream of work keeps both up. At an 18 GiB budget, code decode at 1K context went from 9.3 tok/s to 13.6 with the
helper. In an A/B on that earlier build, helper off against on, decode measured about 7.5 against 13.6 tok/s at an
equal hit rate. The child runs 64x64 elementwise kernels on its own Metal queue: it shares no memory, buffers or MLX state
with the engine, so the engine's logits cannot depend on it.

Control is a pipe: b"1" runs, b"0" pauses (the child then blocks on the pipe, submitting nothing). The child exits on
EOF, so it dies with its parent even while paused (a SIGSTOPped child could outlive a killed engine).
"""

from __future__ import annotations

import os
import subprocess
import sys

CHILD = r"""
import os, select, sys
import mlx.core as mx
fd = int(sys.argv[1])
b = mx.random.normal((64, 64)); mx.eval(b)
run = False
while True:
    r, _, _ = select.select([fd], [], [], None if not run else 0)
    if r:
        c = os.read(fd, 64)
        if not c:
            sys.exit(0)
        run = c[-1:] == b"1"
        continue
    z = b
    for _ in range(64):
        z = z * 1.0001
    mx.eval(z)
"""


def enabled(offload: bool) -> bool:
    """FLINT_GPU_KEEPALIVE=1/0 forces it; unset, it runs in offload mode only."""
    v = os.environ.get("FLINT_GPU_KEEPALIVE", "").strip()
    if v in ("0", "1"):
        return v == "1"
    return offload


class KeepAlive:
    def __init__(self, log=print):
        r, w = os.pipe()
        self._w = w
        self.proc = subprocess.Popen(
            [sys.executable, "-I", "-c", CHILD, str(r)],
            pass_fds=(r,),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os.close(r)
        self.running = False
        log(f"[keepalive] GPU clock keep-alive pid {self.proc.pid} (paused until a job runs)")

    def _send(self, c: bytes) -> None:
        try:
            os.write(self._w, c)
        except OSError:
            pass  # the child is gone: decode is only slower, never wrong

    def resume(self) -> None:
        if not self.running:
            self._send(b"1")
            self.running = True

    def pause(self) -> None:
        if self.running:
            self._send(b"0")
            self.running = False

    def close(self) -> None:
        if self._w is None:
            return
        os.close(self._w)  # EOF: the child exits
        self._w = None
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
