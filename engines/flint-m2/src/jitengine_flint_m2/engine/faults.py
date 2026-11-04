"""The documented fault-injection seam (tests only).

A request carries `fault = "<kind>@<step>"` (HTTP field `x_fault`); it is honoured only when the engine
was created with allow_fault_injection=True (CLI `--allow-fault-injection`), otherwise the request is
rejected as invalid. `step` is the decode pass index (>= 1) at which the fault fires:

  nonfinite         the logits of that pass are multiplied by NaN inside the graph, so the engine's
                    own per-pass finiteness check is what catches it;
  sync_timeout      the synchronisation of that pass is reported as failed (GPUSyncError);
  alloc_failure     a history allocation failure (MemoryError) at that pass;
  thread_exception  the generation thread raises RuntimeError at that pass;
  kernel_stall      a REAL in-kernel stall (megakernel backend): in that pass threadgroup G - 1 returns before any
                    work (prm[P_STALL]; the kernel's pass_entry), so the other threadgroups' waits on its head flags,
                    task counters and grid barriers time out in the kernel and write their error word, which the host
                    reads at the pass's sync (GPUSyncError naming the wait). `kernel_stall@k:persist` stalls every
                    pass >= k; the passes behind a failed one return at once (they see its error at entry), so either
                    form surfaces after one wait limit (plan.SPIN_DEFAULT: ~3 s).

Every fault ends the request with an explicit error, releases no token at or after the faulting pass,
and resets the session state so the next request runs cold (equal to a fresh process).
"""

from __future__ import annotations

from dataclasses import dataclass

KINDS = ("nonfinite", "sync_timeout", "alloc_failure", "thread_exception", "kernel_stall")
PERSIST = ":persist"  # kernel_stall only: every pass from the step on


class GPUSyncError(RuntimeError):
    """A GPU synchronisation failed: the session is invalid and is reset."""


class NonFiniteLogits(FloatingPointError):
    pass


@dataclass(frozen=True)
class Fault:
    kind: str
    step: int
    persist: bool = False


def parse(spec: str | None) -> Fault | None:
    if spec is None:
        return None
    if not isinstance(spec, str) or "@" not in spec:
        raise ValueError(
            "fault must be '<kind>@<step>' (kernel_stall also '<kind>@<step>:persist')"
        )
    kind, _, step = spec.partition("@")
    if kind not in KINDS:
        raise ValueError(f"unknown fault kind {kind!r}; one of {', '.join(KINDS)}")
    persist = step.endswith(PERSIST)
    if persist:
        if kind != "kernel_stall":
            raise ValueError("':persist' applies to kernel_stall only")
        step = step[: -len(PERSIST)]
    try:
        n = int(step)
    except ValueError:
        raise ValueError("fault step must be an integer >= 1") from None
    if n < 1:
        raise ValueError("fault step must be an integer >= 1")
    return Fault(kind, n, persist)


def fire_host(fault: Fault | None, step: int) -> None:
    """Raise the host-side faults scheduled for this decode pass."""
    if fault is None or fault.step != step:
        return
    if fault.kind == "sync_timeout":
        raise GPUSyncError(f"injected GPU synchronisation failure at decode pass {step}")
    if fault.kind == "alloc_failure":
        raise MemoryError(f"injected history allocation failure at decode pass {step}")
    if fault.kind == "thread_exception":
        raise RuntimeError(f"injected generation-thread exception at decode pass {step}")


def poisons_logits(fault: Fault | None, step: int) -> bool:
    return fault is not None and fault.kind == "nonfinite" and fault.step == step


def stalls(fault: Fault | None, step: int) -> bool:
    """kernel_stall: does decode pass `step` carry the in-kernel stall (prm[P_STALL])?"""
    return (
        fault is not None
        and fault.kind == "kernel_stall"
        and (step == fault.step or (fault.persist and step > fault.step))
    )
