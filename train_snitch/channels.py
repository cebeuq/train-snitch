"""Snitch channels: append-only alerts.jsonl, a human-readable latest.txt, macOS notification, webhook, command hook.

alerts.jsonl is the contract with the bigger agent: one JSON object per line, never rewritten. Fields:
  t, time, status (new|ongoing|worsened|escalated|resolved), severity (info|warn|critical), rule, key, what,
  evidence[], action, since, count, source ("rules" | "judge" | "watcher").
"""
from __future__ import annotations

import json
import os
import subprocess
import urllib.request

from .engine import SEV
from .redact import deep as redact_deep

LOUD = {"new", "escalated", "worsened"}


class Channels:
    def __init__(self, cfg: dict, out_dir: str, log=None):
        self.c = cfg.get("channels") or {}
        self.out = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.jsonl = os.path.join(out_dir, "alerts.jsonl")
        self.latest = os.path.join(out_dir, "latest.txt")
        self.log = log or (lambda *_: None)
        self.dry = False

    def emit(self, recs: list[dict], active: list[dict] | None = None, force: bool = False) -> None:
        """force=True: deliver through every enabled channel regardless of severity (TEST alert, watcher crash)."""
        if not recs:
            return
        recs = [redact_deep(r) for r in recs]
        if self.c.get("jsonl", True):
            with open(self.jsonl, "a") as f:
                for r in recs:
                    f.write(json.dumps(r, default=str) + "\n")
        loud = [r for r in recs if force or (not r.get("quiet") and SEV[r["severity"]] >= SEV["warn"] and
                (r["status"] in LOUD or (r["status"] == "ongoing" and r["severity"] == "critical")))]
        shown = [r for r in recs if force or (not r.get("quiet") and SEV[r["severity"]] >= SEV["warn"])]
        if self.c.get("latest", True) and shown:
            self._latest(shown[-1], active or [])
        for r in loud:
            self._notify(r, force)
            self._webhook(r, force)
            self._command(r, force)

    def _latest(self, r: dict, active: list[dict]) -> None:
        lines = [f"{r['severity'].upper()} ({r['status']}) {r['time']}: {r['what']}"]
        lines += [f"  evidence: {e}" for e in r.get("evidence", [])[:8]]
        if r.get("action"):
            lines.append(f"  suggested action: {r['action']}")
        lines.append(f"  rule {r['rule']}, since {r.get('since')}, seen {r.get('count')}x, source {r.get('source', 'rules')}")
        act = [a for a in active if SEV[a["severity"]] >= SEV["warn"]]
        if act:
            lines.append("")
            lines.append("Still active:")
            lines += [f"  - {a['severity']}: {a['what']}" for a in act]
        tmp = self.latest + ".tmp"
        with open(tmp, "w") as f:
            f.write("\n".join(lines) + "\n")
        os.replace(tmp, self.latest)

    def _ok(self, sect: dict, r: dict) -> bool:
        return SEV[r["severity"]] >= SEV[sect.get("min_severity", "warn")]

    def _notify(self, r, force=False):
        n = self.c.get("notify") or {}
        if self.dry or not n.get("enabled") or not (force or self._ok(n, r)):
            return
        msg = f"{r['severity'].upper()} {r['status']}: {r['what']}"[:220]
        title = n.get("title", "train-snitch")
        # pass text as argv to osascript (no string interpolation into the script itself)
        script = ('on run argv\n display notification (item 1 of argv) with title (item 2 of argv) '
                  'sound name "Glass"\nend run')
        try:
            subprocess.run(["osascript", "-e", script, msg, title], timeout=10, capture_output=True)
        except (OSError, subprocess.TimeoutExpired) as e:
            self.log(f"notify failed: {e}")

    def _webhook(self, r, force=False):
        w = self.c.get("webhook") or {}
        url = w.get("url") or os.environ.get(w.get("url_env") or "TRAIN_SNITCH_WEBHOOK", "")
        if self.dry or not url or not (force or self._ok(w, r)):
            return
        try:
            req = urllib.request.Request(url, data=json.dumps(r, default=str).encode(), method="POST",
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=float(w.get("timeout_s", 10))).read()
        except Exception as e:
            self.log(f"webhook failed: {e}")

    def _command(self, r, force=False):
        c = self.c.get("command") or {}
        if self.dry or not c.get("argv") or not (force or self._ok(c, r)):
            return
        try:
            subprocess.run(c["argv"], input=json.dumps(r, default=str) + "\n", text=True, timeout=float(c.get("timeout_s", 20)),
                           capture_output=True)
        except (OSError, subprocess.TimeoutExpired) as e:
            self.log(f"command hook failed: {e}")
