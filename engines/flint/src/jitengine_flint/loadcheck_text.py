"""The fixed text of the load check's stock-MLX stage (loadcheck.py, stage stock), versioned.

A self-contained piece of Python written for this purpose. The check tokenizes it with the checkpoint's tokenizer and
uses its first PROMPT_TOKENS + FORCED_TOKENS ids: the first part is prefilled with stock MLX ops, the next part is fed
to the megakernel one token per pass and compared with stock MLX's echo of the whole text at the same positions. Any
edit to the text is a new version: the stage's bounds (A, D) were calibrated on this one, and tests/unit/test_loadcheck.py
pins the text's sha256.
"""

P0_VERSION = "p0-v1"

P0_TEXT = '''"""A small job scheduler: jobs with priorities and dependencies, run by a fixed pool of workers."""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field


class CycleError(ValueError):
    """The dependency graph has a cycle, so no order can run every job."""


@dataclass(order=True)
class Job:
    priority: int
    name: str = field(compare=False)
    cost: float = field(default=1.0, compare=False)
    needs: tuple = field(default=(), compare=False)


def topological_order(jobs: dict) -> list:
    """Every job after the jobs it needs; ties broken by priority, then by name."""
    indegree = {name: 0 for name in jobs}
    users = {name: [] for name in jobs}
    for job in jobs.values():
        for dep in job.needs:
            if dep not in jobs:
                raise KeyError(f"job {job.name!r} needs unknown job {dep!r}")
            indegree[job.name] += 1
            users[dep].append(job.name)
    ready = [(jobs[n].priority, n) for n, d in indegree.items() if d == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        _, name = heapq.heappop(ready)
        order.append(name)
        for user in users[name]:
            indegree[user] -= 1
            if indegree[user] == 0:
                heapq.heappush(ready, (jobs[user].priority, user))
    if len(order) != len(jobs):
        stuck = sorted(n for n, d in indegree.items() if d > 0)
        raise CycleError(f"cycle among {stuck}")
    return order


def schedule(jobs: dict, workers: int) -> dict:
    """Simulate the pool: each free worker takes the most urgent job whose needs are done.

    Returns the start and finish time of every job. A job starts no earlier than the latest finish of its needs.
    """
    if workers < 1:
        raise ValueError("need at least one worker")
    topological_order(jobs)
    finish: dict = {}
    start: dict = {}
    pending = set(jobs)
    free_at = [0.0] * workers
    tick = itertools.count()
    while pending:
        now = min(free_at)
        worker = free_at.index(now)
        runnable = [jobs[n] for n in pending if all(d in finish and finish[d] <= now for d in jobs[n].needs)]
        if not runnable:
            later = [finish[d] for n in pending for d in jobs[n].needs if d in finish and finish[d] > now]
            free_at[worker] = min(later) if later else now + 1.0
            continue
        job = min(runnable, key=lambda j: (j.priority, j.name, next(tick)))
        pending.discard(job.name)
        start[job.name] = now
        finish[job.name] = now + job.cost
        free_at[worker] = finish[job.name]
    return {name: (start[name], finish[name]) for name in jobs}


def critical_path(jobs: dict) -> tuple:
    """The longest chain of needs by total cost: (cost, [names from first to last])."""
    best: dict = {}
    for name in topological_order(jobs):
        job = jobs[name]
        prev = max(((best[d][0], d) for d in job.needs), default=(0.0, None))
        best[name] = (prev[0] + job.cost, prev[1])
    end = max(best, key=lambda n: best[n][0])
    path = [end]
    while best[path[-1]][1] is not None:
        path.append(best[path[-1]][1])
    return best[end][0], path[::-1]


if __name__ == "__main__":
    demo = {
        "fetch": Job(1, "fetch", 2.0),
        "parse": Job(2, "parse", 1.5, ("fetch",)),
        "index": Job(3, "index", 4.0, ("parse",)),
        "thumbs": Job(5, "thumbs", 3.0, ("fetch",)),
        "report": Job(4, "report", 1.0, ("index", "thumbs")),
    }
    for name, (t0, t1) in sorted(schedule(demo, workers=2).items(), key=lambda kv: kv[1]):
        print(f"{name:8s} {t0:5.1f} -> {t1:5.1f}")
    print("critical path", critical_path(demo))
'''
