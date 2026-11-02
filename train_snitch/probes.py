"""Read-only probes: external commands run on their own minimum interval with a hard timeout.

The watcher never runs anything that changes the run. Probes are argv lists (no shell) unless the adapter
explicitly builds a `bash -c` argv for a read-only pipeline. A probe that is not due this cycle is simply
absent from the results; rules then hold their state.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Probe:
    name: str
    argv: list | None = None
    every_s: float = 60
    timeout_s: float = 60
    parse: str = "text"            # text | lines | json | json_last_line
    argv_fn: Callable | None = None  # (state) -> argv, for incremental probes
    on_result: Callable | None = None  # (result, state) -> None
    stdin: str | None = None
    env: dict | None = None
    ok_rc: tuple = (0,)
    description: str = ""
    cwd: str | None = None
    on_demand: bool = False          # never run by the schedule; only when the adapter calls runner.run() itself


@dataclass
class Result:
    name: str
    t: float
    ok: bool
    rc: int | None
    data: Any = None
    stderr_tail: list = field(default_factory=list)
    secs: float = 0.0


class ProbeRunner:
    def __init__(self, probes: list[Probe], state: dict | None = None, log: Callable | None = None):
        self.probes = {p.name: p for p in probes}
        self.state = state if state is not None else {}
        self.last = self.state.setdefault("probe_last", {})
        self.log = log or (lambda *_: None)

    def due(self, now: float) -> list[Probe]:
        return [p for p in self.probes.values() if not p.on_demand and now - float(self.last.get(p.name, 0)) >= p.every_s]

    def run(self, p: Probe, now: float | None = None) -> Result:
        now = now or time.time()
        self.last[p.name] = now
        argv = p.argv_fn(self.state) if p.argv_fn else p.argv
        t0 = time.time()
        try:
            cp = subprocess.run(argv, input=p.stdin, capture_output=True, text=True, timeout=p.timeout_s,
                                env={**os.environ, **(p.env or {})}, cwd=p.cwd, stdin=None if p.stdin is not None else subprocess.DEVNULL)
            out, err, rc = cp.stdout, cp.stderr, cp.returncode
        except subprocess.TimeoutExpired as e:
            out, err, rc = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or ""), "timeout", None
        except OSError as e:
            out, err, rc = "", f"{e}", None
        data = _parse(out, p.parse)
        res = Result(p.name, now, rc in p.ok_rc and data is not None, rc, data,
                     [x for x in (err or "").splitlines() if x.strip()][-6:], round(time.time() - t0, 2))
        if p.on_result:
            try:
                p.on_result(res, self.state)
            except Exception as ex:
                self.log(f"probe {p.name} on_result failed: {ex!r}")
        return res

    def run_due(self, now: float) -> dict[str, Result]:
        return {p.name: self.run(p, now) for p in self.due(now)}


def _parse(out: str, how: str):
    if how == "text":
        return out
    if how == "lines":
        return out.splitlines()
    if how == "json":
        try:
            return json.loads(out)
        except ValueError:
            return None
    if how == "json_last_line":
        for ln in reversed(out.strip().splitlines()):
            try:
                return json.loads(ln)
            except ValueError:
                continue
        return None
    raise ValueError(how)
