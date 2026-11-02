"""The finite watch loop: probes -> adapter -> rules -> dedupe -> (judge) -> channels, once per cycle.

Lifecycle: exits when the adapter says the run is over, at the deadline (config max_hours / deadline_epoch /
adapter deadline, whichever is first), or on SIGTERM/SIGINT. No daemons, no restarts. One watcher per out_dir
(pid lock). State (alert book, small histories, judge budget) persists in out_dir/state.json so a manual
restart neither re-spams old alerts nor resets the LLM budget.
"""
from __future__ import annotations

import json
import os
import signal
import sys
import time

from . import config as cfgmod
from . import rules as rulemod
from . import timeutil
from . import digest as digestmod
from .channels import Channels
from .engine import Engine, SEV
from .judge import Judge
from .probes import ProbeRunner
from .redact import deep as redact_deep
from .timeutil import fmt12

STOP = {"flag": False, "why": ""}


def _sig(signum, _frame):
    STOP["flag"], STOP["why"] = True, f"signal {signum}"


class Watcher:
    def __init__(self, cfg: dict, rules_only: bool = False, dry: bool = False):
        self.cfg = cfg
        timeutil.set_tz(cfg.get("timezone"))
        rulemod.set_vars(cfg.get("vars"))
        self.out = cfg["out_dir"]
        os.makedirs(self.out, exist_ok=True)
        self.logf = open(os.path.join(self.out, "watcher.log"), "a")
        self.state_path = os.path.join(self.out, "state.json")
        st = {}
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path) as f:
                    st = json.load(f)
                if st.get("name") != cfg.get("name"):
                    st = {}
            except ValueError:
                st = {}
        if rules_only:
            cfg["judge"]["enabled"] = False
        self.adapter = cfgmod.load_adapter(cfg)
        ident = str((self.adapter.summary() or {}).get("run_id") or "")
        if st.get("identity", ident) != ident:
            st = {}   # a different run in the same out_dir: fresh alert book, fresh LLM budget
        self.state = st
        self.state["name"], self.state["identity"] = cfg.get("name"), ident
        self.engine = Engine(rulemod.enabled(cfg, self.adapter.rules()), cfg, st.get("engine"))
        self.runner = ProbeRunner(self.adapter.probes(), self.state.setdefault("probes", {}), log=self.log)
        self.judge = Judge(cfg, self.state, log=self.log, out_dir=self.out)
        self.deadline_t = None
        self.ch = Channels(cfg, self.out, log=self.log)
        self.ch.dry = dry
        self.recent: list = []
        self.cycle_n = int(self.state.get("cycle_n", 0))

    def log(self, msg: str) -> None:
        self.logf.write(f"{timeutil.stamp(time.time())} {msg}\n")
        self.logf.flush()

    # ------------------------------------------------------------------ lock
    def lock(self) -> None:
        p = os.path.join(self.out, "watcher.pid")
        if os.path.exists(p):
            try:
                old = int(open(p).read().strip())
                os.kill(old, 0)
                if old != os.getpid():
                    raise SystemExit(f"another watcher (pid {old}) already writes to {self.out}")
            except (ValueError, ProcessLookupError, PermissionError):
                pass
        with open(p, "w") as f:
            f.write(str(os.getpid()))

    def unlock(self) -> None:
        try:
            os.remove(os.path.join(self.out, "watcher.pid"))
        except OSError:
            pass

    # ------------------------------------------------------------------ helpers
    def deadline(self, start: float) -> float:
        lc = self.cfg.get("lifecycle") or {}
        c = [start + float(lc.get("max_hours", 12)) * 3600]
        if lc.get("deadline_epoch"):
            c.append(float(lc["deadline_epoch"]))
        if lc.get("deadline_local"):        # e.g. "2026-11-01 08:00 AM" in the config's time zone
            d = timeutil.parse_local(lc["deadline_local"])
            if d:
                c.append(d)
        ad = getattr(self.adapter, "deadline", None)
        if ad:
            d = ad()
            if d:
                c.append(float(d))
        return min(c)

    def wrec(self, sev, what, rule, evidence=None, action="") -> dict:
        t = time.time()
        return {"t": t, "time": fmt12(t, date=True), "status": "new", "severity": sev, "rule": rule, "key": rule,
                "what": what, "evidence": evidence or [], "action": action, "source": "watcher",
                "since": fmt12(t, date=True), "count": 1}

    def emit(self, recs, active=None, force=False):
        recs = [redact_deep(r) for r in recs]
        for r in recs:
            r.setdefault("source", "rules")
        self.recent = (self.recent + recs)[-40:]
        self.ch.emit(recs, active, force=force)

    def ctx(self, t: float, view: dict, recs: list, active: list) -> dict:
        h = self.engine.hist
        js = self.state.get("judge") or {}
        text, changed = digestmod.build(t, view, h, recs, active, self.adapter.summary(), js.get("last_call_t") or 0,
                                        js.get("last_step"), self.deadline_t, self.cycle_n)
        data = {"logs": {k: list(v) for k, v in h.log_tail.items()},
                "series": {"losses": list(h.losses), "val": {k: list(v) for k, v in h.val.items()}, "credit": list(h.credit),
                           "gpu": {k: [x[:2] for x in v] for k, v in h.gpu.items()}, "error_lines": list(h.error_lines)},
                "alerts": [{k: r.get(k) for k in ("time", "status", "severity", "key", "what", "action")} for r in self.recent[-20:]]}
        return {"digest_text": text, "changed": changed, "step": h.last_train_step, "data": data,
                "probes": self.adapter.judge_probes()}

    def save(self, t, active, deadline, extra=None):
        self.state["engine"] = self.engine.state()
        self.state["cycle_n"] = self.cycle_n
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.state, f, default=str)
        os.replace(tmp, self.state_path)
        status = {"name": self.cfg.get("name"), "pid": os.getpid(), "updated": fmt12(t, date=True, seconds=True), "t": t,
                  "cycle": self.cycle_n, "deadline": fmt12(deadline, date=True), "judge": self.judge.budget_line(),
                  "judge_mode": ("off: " + self.state["judge"]["rules_only"]) if self.state["judge"].get("rules_only")
                  else f"every {int(float(self.cfg['judge'].get('every_s', 60)))} s ({self.cfg['judge'].get('model')})",
                  "judge_last": self.state["judge"].get("last_verdict"),
                  "active": [dict(a, since=fmt12(a["since"], date=True)) for a in active], **(extra or {})}
        tmp = os.path.join(self.out, "status.json.tmp")
        with open(tmp, "w") as f:
            json.dump(status, f, indent=1, default=str)
        os.replace(tmp, os.path.join(self.out, "status.json"))

    # ------------------------------------------------------------------ one cycle
    def cycle(self, t: float):
        """One cycle. A failure in any stage (adapter, rules, judge) is reported as a deduped warn and the
        watcher keeps going; only `lifecycle.max_cycle_errors` failed cycles in a row stop it."""
        results, view, recs, jrecs, active, errs = {}, {}, [], [], [], []
        try:
            results = self.runner.run_due(t)
            for n, r in results.items():
                self.log(f"probe {n}: ok={r.ok} rc={r.rc} {r.secs}s" + (f" err={r.stderr_tail[-1][:160]}" if r.stderr_tail and not r.ok else ""))
            raw = self.adapter.collect(results, self.engine.hist, self.runner)
            view = self.adapter.normalize(raw, self.engine.hist)
            recs, active = self.engine.process(t, view)
            self.emit(recs, active)
        except Exception as ex:   # noqa: BLE001 - a broken adapter must not kill the watcher silently
            errs.append(("collect", ex))
            active = self._active()
        if not errs:
            try:
                ctx = self.ctx(t, view, recs, active)
                self._write_digest(t, ctx)
                jrecs = self.judge.maybe(t, recs, active, ctx)
                if jrecs:
                    self.emit(jrecs, active)
            except Exception as ex:   # noqa: BLE001
                errs.append(("judge", ex))
        self.cycle_n += 1
        self._cycle_errors(t, errs, active)
        return view, recs + jrecs, active, results

    def _write_digest(self, t, ctx):
        """digest.txt: the compact digest the judge reads this cycle (human-readable; overwritten each cycle)."""
        tmp = os.path.join(self.out, "digest.txt.tmp")
        with open(tmp, "w") as f:
            f.write(f"# digest at {fmt12(t, date=True, seconds=True)} (cycle {self.cycle_n + 1})\n{redact_deep(ctx.get('digest_text') or '')}\n")
        os.replace(tmp, os.path.join(self.out, "digest.txt"))

    def _active(self):
        return [dict(key=e.key, rule=e.rule, severity=e.severity, kind=e.kind, what=e.what, since=e.first_t, count=e.count)
                for e in self.engine.book.values() if e.kind != "milestone"]

    def _cycle_errors(self, t, errs, active):
        import traceback
        es = self.state.setdefault("cycle_errors", {"in_row": 0, "alerted": False})
        if not errs:
            if es["alerted"]:
                self.emit([dict(self.wrec("info", "Resolved: the watcher's cycle works again", "watcher_cycle_error"),
                                status="resolved")], active)
            es["in_row"], es["alerted"] = 0, False
            return
        es["in_row"] += 1
        ev = []
        for stage, ex in errs:
            tb = traceback.format_exception(type(ex), ex, ex.__traceback__)
            self.log(f"cycle error in {stage}: {''.join(tb)[-2000:]}")
            ev.append(f"{stage}: {ex!r}"[:300])
            ev += [x.strip()[:200] for x in tb[-3:-1]]
        lim = int((self.cfg.get("lifecycle") or {}).get("max_cycle_errors", 10))
        if not es["alerted"]:
            es["alerted"] = True
            self.emit([self.wrec("warn", f"Watcher cycle failed ({errs[0][0]}: {type(errs[0][1]).__name__}); "
                                 f"it keeps trying and stops after {lim} failed cycles in a row", "watcher_cycle_error", ev,
                                 "Main agent: read watcher.log in the out dir; the run itself is not affected.")], active)
        if es["in_row"] >= lim:
            raise _GiveUp(f"{lim} failed cycles in a row, last {errs[-1][0]}: {errs[-1][1]!r}"[:300])

    def run(self, once: bool = False, test_alert: bool = False) -> str:
        signal.signal(signal.SIGTERM, _sig)
        signal.signal(signal.SIGINT, _sig)
        self.lock()
        start = time.time()
        reason = "crashed: unknown"
        deadline = start
        try:
            deadline = self.deadline_t = self.deadline(start)
            jm = "off" if not self.cfg["judge"].get("enabled") else \
                f"{self.cfg['judge'].get('model')} every {int(float(self.cfg['judge'].get('every_s', 60)))} s, credit cap {self.cfg['judge'].get('credit_cap', 150)}"
            self.emit([self.wrec("info", f"watcher started ({'one cycle' if once else 'until ' + fmt12(deadline, date=True)}; judge {jm})",
                                 "watcher_start", [f"out dir {self.out}", f"adapter {self.adapter.name}"])])
            if test_alert:
                self.emit([test_record(self.cfg)], force=True)
            self.emit(self.judge.start())
            while True:
                t = time.time()
                deadline = self.deadline_t = self.deadline(start)   # the adapter's deadline may appear later (run created)
                if t >= deadline:
                    reason = f"deadline {fmt12(deadline, date=True)} reached"
                    break
                view, recs, active, results = self.cycle(t)
                self.log(f"cycle {self.cycle_n}: probes {sorted(results)}; {len(recs)} records; active "
                         + ", ".join(f"{a['key']}={a['severity']}" for a in active))
                self.save(t, active, deadline)
                d = self.adapter.done(view, self.engine.hist, t) if view else None
                if d:
                    reason = f"run is over: {d}"
                    break
                if once:
                    reason = "single cycle (--once)"
                    break
                nxt = t + float(self.cfg.get("cycle_s", 60))
                while time.time() < nxt and not STOP["flag"]:
                    time.sleep(min(1.0, max(0.0, nxt - time.time())))
                if STOP["flag"]:
                    reason = f"stopped by {STOP['why']}"
                    break
            if self.state["judge"].get("calls"):
                m = self.judge.meter_fn()
                if m:
                    self.judge._credit_check(m.get("credits"))
            return reason
        except _GiveUp as ex:
            reason = f"crashed: {ex}"
            return reason
        except BaseException as ex:   # noqa: BLE001 - always leave an exit record for `tail`
            reason = f"crashed: {ex!r}"[:300]
            self.log(f"watcher crashed: {ex!r}")
            raise
        finally:
            sev = "warn" if reason.startswith("crashed") else "info"
            try:
                self.emit([self.wrec(sev, f"watcher exiting: {reason}", "watcher_exit", [self.judge.budget_line()],
                                     "Main agent: the watcher is gone; restart it (start.sh) if the run is still live."
                                     if sev == "warn" else "")], force=sev == "warn")
                self.save(time.time(), self._active(), deadline, {"exited": reason})
            except Exception as ex:   # noqa: BLE001
                self.log(f"exit bookkeeping failed: {ex!r}")
            self.unlock()
            self.logf.close()


class _GiveUp(Exception):
    pass


def test_record(cfg: dict) -> dict:
    """One clearly labeled TEST alert (info) that every channel delivers, to prove the wiring end to end."""
    t = time.time()
    return {"t": t, "time": fmt12(t, date=True), "status": "new", "severity": "info", "rule": "test", "key": "test",
            "what": f"TEST alert from train-snitch ({cfg.get('name')}): delivery check only, nothing is wrong",
            "evidence": ["TEST: sent once at watcher start with --test-alert"], "action": "None: this is a TEST.",
            "source": "watcher", "since": fmt12(t, date=True), "count": 1, "test": True}
