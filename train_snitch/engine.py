"""Rule engine: turns one normalized snapshot ("view") per cycle into findings, then dedupes them into
alert records with a status: new / ongoing / worsened / escalated / resolved.

- A *condition* finding (e.g. "disk low") stays active while the rule keeps reporting it and resolves when the
  rule, having readable input, stops reporting it for `clear_cycles` cycles (hysteresis).
- An *event* finding (e.g. one OOM log line) stays active for `event_ttl_s` after it was last seen; repeats
  inside that window are ongoing, not new. Events never emit "resolved".
- A *milestone* finding (e.g. "smoke test passed") is emitted once as info.
- A rule that returns None had no readable input this cycle: its active findings are held unchanged.
- While a finding is ongoing it is re-emitted only as a sparse reminder (critical every 30 min, warn every 2 h).
"""
from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Any, Callable

from . import timeutil
from .redact import redact

SEV = {"info": 0, "warn": 1, "critical": 2}


@dataclass
class Finding:
    rule: str
    severity: str
    what: str
    evidence: list = field(default_factory=list)
    action: str = ""
    key: str | None = None
    kind: str = "condition"          # condition | event | milestone
    metric: float | None = None      # for "worsened": larger is worse
    worsen_factor: float | None = None
    escalate_to_judge: bool = False

    def k(self) -> str:
        return self.key or self.rule


class History:
    """Small bounded memory shared by rules (JSON-serializable so a restarted watcher keeps context)."""

    def __init__(self, d: dict | None = None):
        d = d or {}
        self.losses = deque(d.get("losses", []), maxlen=200)          # [step, loss]
        self.last_train_step = d.get("last_train_step")
        self.val = {k: deque(v, maxlen=50) for k, v in d.get("val", {}).items()}   # kind -> [[upd, loss]]
        self.val_seen = set(tuple(x) for x in d.get("val_seen", []))
        self.val_baseline = d.get("val_baseline", {})
        self.gpu = {k: deque(v, maxlen=60) for k, v in d.get("gpu", {}).items()}   # idx -> [[t, util, line]]
        self.credit = deque(d.get("credit", []), maxlen=200)          # [t, usd]
        self.first_seen = d.get("first_seen", {})
        self.series = {k: deque(v, maxlen=50) for k, v in d.get("series", {}).items()}
        self.pulled = set(d.get("pulled", []))
        self.counters = d.get("counters", {})
        self.last_step = d.get("last_step")
        self.error_lines = deque(d.get("error_lines", []), maxlen=60)  # [t, src, line]
        self.log_tail = {k: deque(v, maxlen=80) for k, v in d.get("log_tail", {}).items()}
        self.new_train_rows: list = []
        self.new_val_rows: list = []

    def seen(self, name: str, t: float) -> float:
        return self.first_seen.setdefault(name, t)

    def forget(self, name: str) -> None:
        self.first_seen.pop(name, None)

    def to_json(self) -> dict:
        return {"losses": list(self.losses), "last_train_step": self.last_train_step,
                "val": {k: list(v) for k, v in self.val.items()}, "val_seen": [list(x) for x in self.val_seen],
                "val_baseline": self.val_baseline, "gpu": {k: list(v) for k, v in self.gpu.items()},
                "credit": list(self.credit), "first_seen": self.first_seen,
                "series": {k: list(v) for k, v in self.series.items()}, "pulled": sorted(self.pulled),
                "counters": self.counters, "last_step": self.last_step, "error_lines": list(self.error_lines),
                "log_tail": {k: list(v) for k, v in self.log_tail.items()}}


@dataclass
class Entry:
    key: str
    rule: str
    source: str
    severity: str
    kind: str
    what: str
    first_t: float
    last_seen_t: float
    last_emit_t: float
    count: int = 1
    metric_base: float | None = None
    clear_count: int = 0
    action: str = ""
    evidence: list = field(default_factory=list)


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


class Engine:
    def __init__(self, rules: list[Callable], cfg: dict, state: dict | None = None):
        self.rules = rules
        self.cfg = cfg
        state = state or {}
        self.hist = History(state.get("hist"))
        self.book: dict[str, Entry] = {k: Entry(**v) for k, v in state.get("book", {}).items()}
        e = cfg.get("engine", {})
        self.clear_cycles = int(e.get("clear_cycles", 1))
        self.event_ttl = float(e.get("event_ttl_s", 2700))
        self.remind = {"critical": float(e.get("remind_critical_s", 1800)), "warn": float(e.get("remind_warn_s", 7200))}
        self.evidence_res = [_re(p) for p in cfg.get("evidence_patterns", [])]
        self.benign_res = [_re(p) for p in cfg.get("benign_patterns", [])]

    # ---- ingestion shared by all rules -------------------------------------------------------------
    def ingest(self, t: float, v: dict) -> None:
        h = self.hist
        for src, lines in (v.get("log_lines") or {}).items():
            tail = h.log_tail.setdefault(src, deque(maxlen=80))
            for ln in lines or []:
                ln = redact(ln) if isinstance(ln, str) else ln
                tail.append(ln)
                if any(r.search(ln) for r in self.evidence_res) and not any(r.search(ln) for r in self.benign_res):
                    h.error_lines.append([t, src, ln[:400]])
        # new train rows (dedupe by step so overlapping sources don't double count)
        h.new_train_rows = []
        for r in v.get("train_rows") or []:
            st = r.get("step")
            if st is None or h.last_train_step is None or st > h.last_train_step:
                h.new_train_rows.append(r)
                if st is not None:
                    h.last_train_step = st
        h.new_val_rows = []
        for r in v.get("val_rows") or []:
            sig = (r.get("kind") or "val", r.get("step"))
            if sig in h.val_seen:
                continue
            h.val_seen.add(sig)
            h.new_val_rows.append(r)
            if _finite(r.get("loss")):  # val series include this cycle's rows (rules compare the newest point)
                kind = r.get("kind") or "val"
                h.val.setdefault(kind, deque(maxlen=50)).append([r.get("step"), r["loss"]])
                if r.get("step") == 0 and kind not in h.val_baseline:
                    h.val_baseline[kind] = r["loss"]
        for s in v.get("gpu_samples") or []:
            h.gpu.setdefault(str(s[1]), deque(maxlen=60)).append([s[0], s[2], s[4] if len(s) > 4 else ""])
        if _finite(v.get("credit_usd")):
            if not h.credit or h.credit[-1][0] < t:
                h.credit.append([t, float(v["credit_usd"])])

    def post_ingest(self) -> None:
        h = self.hist
        for r in h.new_train_rows:
            if _finite(r.get("loss")):
                h.losses.append([r.get("step"), r["loss"]])

    # ---- one cycle -----------------------------------------------------------------------------------
    def process(self, t: float, view: dict) -> tuple[list[dict], list[dict]]:
        """Returns (records to emit, snapshot of every active entry after this cycle)."""
        self.ingest(t, view)
        results: dict[str, list[Finding] | None] = {}
        for rule in self.rules:
            name = getattr(rule, "rule_name", rule.__name__)
            try:
                res = rule(t, view, self.hist, self.cfg)
            except Exception as ex:  # a broken rule must never kill the watcher
                res = [Finding("watcher_rule_error", "info", f"rule {name} crashed: {ex!r}", kind="event",
                               key=f"rule_error:{name}")]
            results[name] = res
        self.post_ingest()
        records: list[dict] = []
        seen_keys: set[str] = set()
        for src, res in results.items():
            if res is None:
                continue
            merged: dict[str, Finding] = {}
            for f in res:
                k = f.k()
                if k in merged:  # same key twice in one cycle: keep the worse one, merge evidence
                    o = merged[k]
                    if SEV[f.severity] > SEV[o.severity]:
                        f.evidence = _uniq(f.evidence + o.evidence)
                        merged[k] = f
                    else:
                        o.evidence = _uniq(o.evidence + f.evidence)
                else:
                    merged[k] = f
            for k, f in merged.items():
                seen_keys.add(k)
                rec = self._update(t, src, f)
                if rec:
                    records.append(rec)
        # absent keys: resolve conditions / expire events, but only for rules that had readable input
        for k, e in list(self.book.items()):
            if k in seen_keys:
                continue
            if e.kind == "milestone":
                continue
            if e.kind == "event":
                if t - e.last_seen_t > self.event_ttl:
                    del self.book[k]
                continue
            if results.get(e.source, "missing") is None:
                continue  # input unreadable: hold
            if e.source not in results:
                continue
            e.clear_count += 1
            if e.clear_count >= self.clear_cycles:
                del self.book[k]
                if SEV[e.severity] >= SEV["warn"]:
                    records.append(self._record(t, e, "resolved", severity="info",
                                                what=f"Resolved: {e.what}", evidence=[], action=""))
        snap = [dict(key=e.key, rule=e.rule, severity=e.severity, kind=e.kind, what=e.what,
                     since=e.first_t, count=e.count) for e in self.book.values() if e.kind != "milestone"]
        return records, snap

    def _update(self, t: float, src: str, f: Finding) -> dict | None:
        f.what, f.action = redact(f.what), redact(f.action)
        f.evidence = [redact(x) if isinstance(x, str) else x for x in (f.evidence or [])]
        k = f.k()
        e = self.book.get(k)
        if e is None or (e.kind == "milestone" and f.kind != "milestone"):
            e = Entry(key=k, rule=f.rule, source=src, severity=f.severity, kind=f.kind, what=f.what, first_t=t,
                      last_seen_t=t, last_emit_t=t, metric_base=f.metric, action=f.action, evidence=f.evidence)
            self.book[k] = e
            return self._record(t, e, "new", f=f)
        if e.kind == "milestone":
            return None
        e.last_seen_t, e.clear_count, e.count = t, 0, e.count + 1
        e.what, e.action, e.evidence, e.rule = f.what, f.action, f.evidence, f.rule
        if SEV[f.severity] > SEV[e.severity]:
            e.severity, e.last_emit_t = f.severity, t
            if f.metric is not None:
                e.metric_base = f.metric
            return self._record(t, e, "escalated", f=f)
        if SEV[f.severity] < SEV[e.severity]:
            e.severity = f.severity  # quietly de-escalate; a later rise is reported again
            return None
        if (f.metric is not None and f.worsen_factor and e.metric_base and e.metric_base > 0
                and f.metric >= f.worsen_factor * e.metric_base):
            e.metric_base, e.last_emit_t = f.metric, t
            return self._record(t, e, "worsened", f=f)
        gap = self.remind.get(e.severity)
        if gap and t - e.last_emit_t >= gap:
            e.last_emit_t = t
            return self._record(t, e, "ongoing", f=f)
        return None

    def _record(self, t, e: Entry, status, f: Finding | None = None, **over) -> dict:
        r = {"t": t, "time": timeutil.fmt12(t, date=True), "status": status, "severity": e.severity,
             "rule": e.rule, "key": e.key, "what": e.what, "evidence": list(e.evidence)[:12],
             "action": e.action, "since": timeutil.fmt12(e.first_t, date=True), "count": e.count,
             "escalate_to_judge": bool(f and f.escalate_to_judge)}
        r.update(over)
        return r

    def state(self) -> dict:
        return {"hist": self.hist.to_json(), "book": {k: asdict(v) for k, v in self.book.items()}}


def _uniq(xs):
    out, s = [], set()
    for x in xs:
        if x not in s:
            s.add(x)
            out.append(x)
    return out


def _re(p):
    import re
    return re.compile(p)


def median(xs):
    return statistics.median(xs) if xs else None
