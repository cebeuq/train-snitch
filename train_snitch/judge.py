"""LLM judge: a small model (default GPT-6 Luna through the Pi harness) reads a compact digest + the deltas
since its last look, every `every_s` (default 60 s), and decides: ok / watch / snitch.

- ok      nothing to report (logged to judge.jsonl only)
- watch   something to keep an eye on (judge.jsonl + status.json only; no alert)
- snitch  report to the bigger agent: an alert record (source "judge"), deduped by the model's `topic` slug:
          new topic -> "new"; same topic at higher severity -> "escalated"; repeats inside `snitch_repeat_min`
          are swallowed, after that one "ongoing" reminder per window.
When the rule layer raised a new warn+ alert this cycle, the judgment runs as a Pi AgentSession with read-only tools
(`agent_on_anomaly`); otherwise it is one stateless ModelRuntime.complete() call.

Guards (rules never stop; the judge is advisory and never suppresses a rule alert):
- credit cap: Codex credits are read at start (codex app-server, no model call), from each response's
  x-codex-credits-balance header, and every `meter_every_calls`. When the balance has fallen more than `credit_cap`
  since the watcher started (default 150), the judge turns off for the rest of the run: one warn alert, rules-only.
  The balance is account-wide, so other Codex use in the same window counts too (conservative).
- `max_consecutive_errors` failed calls in a row (default 3), or a provider usage-limit refusal: same back-off.
- `max_calls` model turns per run: hard backstop.
- `cost_cap_usd` (any provider): Pi's own per-call USD cost summed over the run; above it -> same back-off.
Model choice (any Pi provider, custom endpoints, fallbacks) is validated and mapped in `llm.py`.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time

from . import llm
from . import meter as codex_meter
from .redact import deep as redact_deep, redact
from .engine import SEV
from .timeutil import fmt12

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PI_PKG_NAME = "@earendil-works/pi-coding-agent"


def asset_dir() -> str:
    """judge/ in a checkout; train_snitch/_judge/ (the same files) in an installed wheel."""
    d = os.path.join(REPO, "judge")
    if os.path.isfile(os.path.join(d, "pi_judge.mjs")):
        return d
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "_judge")


def find_pi_pkg(explicit: str | None = None) -> str:
    if explicit:
        return os.path.expanduser(explicit)
    if os.environ.get("PI_PKG"):
        return os.environ["PI_PKG"]
    try:
        root = subprocess.run(["npm", "root", "-g"], capture_output=True, text=True, timeout=20).stdout.strip()
        if root and os.path.isdir(os.path.join(root, PI_PKG_NAME)):
            return os.path.join(root, PI_PKG_NAME)
    except (OSError, subprocess.TimeoutExpired):
        pass
    return os.path.join("/opt/homebrew/lib/node_modules", PI_PKG_NAME)


class Judge:
    def __init__(self, cfg: dict, state: dict, log=None, meter_fn=None, bridge_fn=None, out_dir: str | None = None):
        self.cfg = cfg
        self.j = cfg.get("judge") or {}
        self.log = log or (lambda *_: None)
        self.out_dir = out_dir
        self.st = state.setdefault("judge", {})
        for k, v in {"calls": 0, "judgments": 0, "errors_in_row": 0, "last_call_t": 0, "start_credits": None,
                     "last_credits": None, "rules_only": None, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0,
                     "meter": None, "probe_last_t": 0, "topics": {}, "last_verdict": None, "last_step": None,
                     "decisions": {"ok": 0, "watch": 0, "snitch": 0}}.items():
            self.st.setdefault(k, v)
        self.meter_fn = meter_fn or (codex_meter.read_summary if llm.meter_kind(self.j) == "codex" else (lambda: None))
        self.bridge_fn = bridge_fn or self._bridge
        with open(os.path.join(asset_dir(), "system-prompt.md")) as f:
            self.system_prompt = f.read()
        self._pi = None

    # ------------------------------------------------------------------ guard
    def start(self) -> list[dict]:
        if not self.j.get("enabled"):
            self.st["rules_only"] = None
            return self._off("disabled (rules-only mode)", "info")
        # a (re)started watcher gets a fresh error count; the credit cap is per run and is re-checked right here
        self.st["rules_only"], self.st["errors_in_row"] = None, 0
        m = self.meter_fn()
        self.st["meter"] = m
        if m and m.get("credits") is not None:
            if self.st.get("start_credits") is None:
                self.st["start_credits"] = m["credits"]
            return self._credit_check(m["credits"])
        if llm.meter_kind(self.j) == "codex":
            self.log("judge: Codex meter unreadable at start; the credit cap uses response headers only")
        return []

    def _off(self, reason: str, sev: str = "warn") -> list[dict]:
        if self.st.get("rules_only"):
            return []
        self.st["rules_only"] = reason
        t = time.time()
        return [{"t": t, "time": fmt12(t, date=True), "status": "new", "severity": sev, "rule": "judge_off", "key": "judge_off",
                 "what": f"LLM judge off ({reason}); the watcher continues rules-only", "evidence": [self.budget_line()],
                 "action": "" if sev == "info" else "Nothing breaks: rule alerts continue every cycle without the LLM.",
                 "source": "watcher", "since": fmt12(t, date=True), "count": 1}]

    def credits_used(self):
        s = self.st
        if s.get("start_credits") is None or s.get("last_credits") is None:
            return None
        return s["start_credits"] - s["last_credits"]

    def budget_line(self) -> str:
        s, used = self.st, self.credits_used()
        return (f"judge calls {s['calls']}/{self.j.get('max_calls', 2000)}, judgments {s['judgments']} "
                f"(ok {s['decisions'].get('ok', 0)}, watch {s['decisions'].get('watch', 0)}, snitch {s['decisions'].get('snitch', 0)}), "
                f"tokens {s['tokens_in']} in / {s['tokens_out']} out, credits used "
                + (f"{used:.3f} of cap {self.j.get('credit_cap', 150)}" if used is not None else "unknown"))

    def _credit_check(self, bal) -> list[dict]:
        if bal is None:
            return []
        self.st["last_credits"] = bal
        if self.st.get("start_credits") is None:
            self.st["start_credits"] = bal
        used = self.st["start_credits"] - bal
        cap = float(self.j.get("credit_cap", 150))
        if used > cap:
            return self._off(f"Codex credits fell {used:.2f} since start, over the per-run cap of {cap:g}")
        return []

    # ------------------------------------------------------------------ one tick
    def maybe(self, t: float, recs: list[dict], active: list[dict], ctx: dict) -> list[dict]:
        if not self.j.get("enabled") or self.st.get("rules_only"):
            return []
        every = float(self.j.get("every_s", 60))
        if t - self.st["last_call_t"] < every - 2:
            return []
        trig = [r for r in recs if r.get("source", "rules") == "rules" and r["status"] in ("new", "escalated", "worsened")
                and (SEV[r["severity"]] >= SEV["warn"] or r.get("escalate_to_judge"))]
        if (self.j.get("skip_unchanged") and not trig and not ctx.get("changed")
                and t - self.st["last_call_t"] < float(self.j.get("max_quiet_min", 15)) * 60):
            return []
        max_calls, turns = int(self.j.get("max_calls", 2000)), int(self.j.get("max_turns", 3))
        left = max_calls - self.st["calls"]
        if left <= 0:
            return self._off(f"call cap {max_calls} reached", "info")
        mode = "agent" if (trig and self.j.get("agent_on_anomaly", True) and left >= turns) else "single"
        out = []
        n = int(self.j.get("meter_every_calls", 15))
        if self.st["calls"] and n and self.st["judgments"] % n == 0:
            m = self.meter_fn()
            if m:
                self.st["meter"] = m
                out += self._credit_check(m.get("credits"))
                if self.st.get("rules_only"):
                    return out
        req = self._request(t, trig, active, ctx, mode, turns if mode == "agent" else 1)
        self.st["last_call_t"] = t
        res = self.bridge_fn(req) or {}
        used = int(res.get("turns") or 1)
        self.st["calls"] += used
        self.st["judgments"] += 1
        u = res.get("usage") or {}
        self.st["tokens_in"] += int(u.get("input") or 0)
        self.st["tokens_out"] += int(u.get("output") or 0)
        self.st["cost_usd"] = round(self.st.get("cost_usd", 0.0) + float(u.get("cost") or 0), 6)
        self._cooldowns(t, res.get("attempts"))
        if res.get("probes_run"):
            self.st["probe_last_t"] = t
        hdr = res.get("headers") or {}
        try:
            if hdr.get("x-codex-credits-balance"):
                out += self._credit_check(float(hdr["x-codex-credits-balance"]))
        except ValueError:
            pass
        err = res.get("error") or ""
        v = res.get("verdict")
        self._journal(t, mode, trig, res)
        cap_usd = self.j.get("cost_cap_usd")
        if cap_usd is not None and self.st["cost_usd"] > float(cap_usd):
            out += self._off(f"judge spend ${self.st['cost_usd']:.4f} (Pi's cost figure) over the per-run cap of ${float(cap_usd):g}")
        if self.st.get("rules_only"):
            return out
        att = res.get("attempts")      # model chain: refused only when every candidate was rate-limited
        if (all(a.get("error_kind") == "rate_limit" for a in att) if att else
                re.search(r"usage_limit|rate_limit|429|insufficient|quota|credits? (?:exhausted|depleted)", err, re.I)):
            return out + self._off(f"provider refused: {err[:120]}")
        if not v:
            self.st["errors_in_row"] += 1
            self.log(f"judge gave no verdict ({self.st['errors_in_row']} in a row): {err[:200]}")
            lim = int(self.j.get("max_consecutive_errors", 3))
            if self.st["errors_in_row"] >= lim:
                return out + self._off(f"{lim} model errors in a row, last: {err[:100]}")
            return out
        self.st["errors_in_row"] = 0
        self.st["last_step"] = ctx.get("step")
        return out + self._apply(t, v, trig, res, mode)

    def _apply(self, t, v, trig, res, mode) -> list[dict]:
        dec = v.get("decision") if v.get("decision") in ("ok", "watch", "snitch") else "watch"
        sev = v.get("severity") if v.get("severity") in SEV else "info"
        self.st["decisions"][dec] = self.st["decisions"].get(dec, 0) + 1
        self.st["last_verdict"] = {"time": fmt12(t, seconds=True), "decision": dec, "severity": sev,
                                   "topic": v.get("topic"), "what": str(v.get("what", ""))[:200]}
        out = []
        if trig and v.get("assessment") == "likely_false_alarm":
            out.append(self._rec(t, "false_alarm", "info", "new",
                                 f"Judge: likely false alarm: {v.get('what', '')}", v, trig, res, mode,
                                 about=[r["key"] for r in trig]))
        if dec != "snitch" or sev == "info":
            return out
        topic = re.sub(r"[^a-z0-9_-]+", "-", str(v.get("topic") or "general").lower()).strip("-")[:40] or "general"
        book = self.st["topics"]
        ttl = float(self.cfg.get("engine", {}).get("event_ttl_s", 2700))
        for k in [k for k, e in book.items() if t - e["last_t"] > max(ttl, float(self.j.get("snitch_repeat_min", 30)) * 60)]:
            del book[k]
        e = book.get(topic)
        if e is None:
            book[topic] = {"sev": sev, "first_t": t, "last_t": t, "emit_t": t, "count": 1}
            status = "new"
        else:
            e["last_t"], e["count"] = t, e["count"] + 1
            if SEV[sev] > SEV[e["sev"]]:
                e["sev"], e["emit_t"], status = sev, t, "escalated"
            elif t - e["emit_t"] >= float(self.j.get("snitch_repeat_min", 30)) * 60:
                e["emit_t"], status = t, "ongoing"
            else:
                return out
            e["sev"] = max(e["sev"], sev, key=lambda s: SEV[s])
        e = book[topic]
        out.append(self._rec(t, topic, e["sev"], status, f"Judge: {v.get('what', '')}", v, trig, res, mode,
                             since=e["first_t"], count=e["count"]))
        return out

    def _rec(self, t, topic, sev, status, what, v, trig, res, mode, about=None, since=None, count=1):
        ev = [str(x)[:300] for x in (v.get("evidence") or [])[:6]]
        ev += [f"assessment: {v.get('assessment')} (confidence {v.get('confidence', '?')}), {mode} call"]
        if trig:
            ev.append("rule alerts this cycle: " + ", ".join(r["key"] for r in trig))
        if res.get("probes_run"):
            ev.append("judge ran: " + ", ".join(res["probes_run"]))
        ev.append(self.budget_line())
        r = {"t": t, "time": fmt12(t, date=True), "status": status, "severity": sev, "rule": f"judge:{topic}",
             "key": f"judge:{topic}", "what": what[:300], "evidence": ev, "action": str(v.get("action", ""))[:300],
             "source": "judge", "since": fmt12(since or t, date=True), "count": count,
             "judge_model": res.get("model_used") or self.j.get("model"),
             "decision": v.get("decision"), "assessment": v.get("assessment")}
        if about:
            r["about"] = about
        return r

    def _journal(self, t, mode, trig, res):
        if not self.out_dir:
            return
        row = {"t": t, "time": fmt12(t, seconds=True), "mode": mode, "trigger": [r["key"] for r in trig],
               "verdict": res.get("verdict"), "error": res.get("error"), "turns": res.get("turns"), "ms": res.get("ms"),
               "usage": res.get("usage"), "credits_used": self.credits_used(), "probes_run": res.get("probes_run")}
        for k in ("model_used", "attempts"):          # only present with a model chain (docs/JUDGE-MODELS.md)
            if res.get(k):
                row[k] = res[k]
        try:
            with open(os.path.join(self.out_dir, "judge.jsonl"), "a") as f:
                f.write(json.dumps(row, default=str) + "\n")
        except OSError:
            pass

    # ------------------------------------------------------------------ request
    def _request(self, t, trig, active, ctx, mode, turns) -> dict:
        lines = [ctx.get("digest_text") or ""]
        if trig:
            lines.append("NEW rule alerts this cycle (the reason this is an anomaly call):")
            for r in trig[:6]:
                lines.append(f"- [{r['severity']}/{r['status']}] {r['key']}: {r['what']}")
                lines += [f"    evidence: {e}"[:300] for e in r.get("evidence", [])[:6]]
                if r.get("action"):
                    lines.append(f"    rule's suggestion: {r['action']}"[:300])
        lv = self.st.get("last_verdict")
        if lv:
            lines.append(f"Your previous verdict ({lv['time']}): {lv['decision']}/{lv['severity']} topic={lv.get('topic')}: {lv['what']}")
        if self.st["topics"]:
            lines.append("Topics you already snitched (reuse the same slug for the same problem): "
                         + ", ".join(f"{k} ({e['sev']}, x{e['count']})" for k, e in self.st["topics"].items()))
        msg = redact("\n".join(x for x in lines if x))[: int(self.j.get("max_prompt_chars", 9000))]
        probes = []
        if mode == "agent" and t - self.st.get("probe_last_t", 0) >= float(self.j.get("probe_gap_min", 15)) * 60:
            probes = [dict(p, argv=[sys.executable, "-I", os.path.join(REPO, "train_snitch", "redact.py"), "--", *p["argv"]])
                      for p in (ctx.get("probes") or [])]   # probe output is redacted before the model sees it
        if self._pi is None:
            self._pi = find_pi_pkg(self.j.get("pi_pkg"))
        req = {"model": self.j.get("model"), "mode": mode, "max_turns": turns, "reasoning_effort": self.j.get("reasoning_effort", "none"),
               "timeout_ms": int(float(self.j.get("timeout_s", 90)) * 1000), "system_prompt": self.system_prompt,
               "user_message": msg, "data": redact_deep(ctx.get("data") or {}) if mode == "agent" else {}, "probes": probes,
               "pi_pkg": self._pi, "max_tokens": int(self.j.get("max_tokens", 500))}
        if llm.uses_candidates(self.j):     # model chain / custom provider: extra keys; without them the 0.1 request
            req["candidates"] = self.candidates(t)
            req["fallback_on"] = list(self.j.get("fallback_on") or llm.DEFAULT_FALLBACK_ON)
        return req

    def candidates(self, t: float) -> list[dict]:
        """Bridge candidates in order, minus those cooling down after a rate limit / missing auth. When every
        candidate is cooling down, all are sent (the cooldown is a hint, never a reason to skip a judgment)."""
        cands = [llm.bridge_candidate(c, self.j) for c in llm.candidates(self.j)]
        cool = self.st.get("cooldown") or {}
        live = [c for c in cands if cool.get(c["model"], 0) <= t]
        return live or cands

    def _cooldowns(self, t, attempts):
        if not attempts:
            return
        mins = float(self.j.get("fallback_cooldown_min") if self.j.get("fallback_cooldown_min") is not None else 15)
        cool = self.st.setdefault("cooldown", {})
        for a in attempts:
            if a.get("error_kind") in ("rate_limit", "no_auth", "not_found") and a.get("model"):
                cool[a["model"]] = t + mins * 60
                self.log(f"judge: {a['model']} {a['error_kind']}; skipped for {mins:g} min while others are left")
        for k in [k for k, until in cool.items() if until <= t]:
            del cool[k]

    def _bridge(self, req: dict) -> dict:
        node = self.j.get("node") or shutil.which("node") or "/opt/homebrew/bin/node"
        bridge = self.j.get("bridge") or os.path.join(asset_dir(), "pi_judge.mjs")
        env = None
        if self.j.get("pi_agent_dir"):      # Pi's own variable for its config dir (auth.json, models.json)
            env = dict(os.environ, PI_CODING_AGENT_DIR=os.path.expanduser(self.j["pi_agent_dir"]))
        try:
            cp = subprocess.run([node, bridge], input=json.dumps(req, default=str), capture_output=True, text=True,
                                timeout=req["timeout_ms"] / 1000 + 30, cwd=REPO, env=env)
        except (OSError, subprocess.TimeoutExpired) as e:
            return {"error": f"bridge: {e}", "turns": 1}
        try:
            return json.loads(cp.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return {"error": f"bridge rc {cp.returncode}: {(cp.stderr or '')[-300:]}", "turns": 1}
