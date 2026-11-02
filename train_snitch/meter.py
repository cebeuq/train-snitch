#!/usr/bin/env python3
"""Codex usage meter. Read the Codex (ChatGPT sign-in) usage meter and credit balance WITHOUT a model call.

Spawns `codex app-server` (stdio JSON-RPC), sends initialize + account/rateLimits/read, prints one JSON line:
{"primary_pct":..,"secondary_pct":..,"credits":..,"has_credits":..,"reached":..,"resets_at":..}
Stdlib only. Never prints tokens.
"""
import json, os, shutil, subprocess, sys, time

CODEX = os.environ.get("CODEX_BIN") or shutil.which("codex") or "/opt/homebrew/bin/codex"


def read_meter(timeout=20.0):
    p = subprocess.Popen([CODEX, "app-server"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    try:
        def send(obj):
            p.stdin.write(json.dumps(obj) + "\n"); p.stdin.flush()
        send({"id": 1, "method": "initialize", "params": {"clientInfo": {"name": "train-snitch", "version": "0.0.1"}}})
        send({"method": "initialized"})
        send({"id": 2, "method": "account/rateLimits/read", "params": None})
        t0 = time.time()
        while time.time() - t0 < timeout:
            line = p.stdout.readline()
            if not line:
                break
            try:
                m = json.loads(line)
            except ValueError:
                continue
            if m.get("id") == 2:
                if "error" in m:
                    raise RuntimeError(m["error"])
                return m["result"]
        raise TimeoutError("no rateLimits response")
    finally:
        p.kill()
        try:
            p.wait(timeout=5)
        except Exception:
            pass


def summarize(r):
    rl = r.get("rateLimits") or r
    prim, sec, cr = rl.get("primary") or {}, rl.get("secondary") or {}, rl.get("credits") or {}
    bal = cr.get("balance")
    return {"primary_pct": prim.get("usedPercent"), "primary_resets_at": prim.get("resetsAt"),
            "secondary_pct": sec.get("usedPercent"), "secondary_resets_at": sec.get("resetsAt"),
            "credits": float(bal) if bal not in (None, "") else None, "has_credits": cr.get("hasCredits"),
            "reached": rl.get("rateLimitReachedType"), "plan": rl.get("planType")}


def read_summary(timeout=20.0):
    """Never raises; returns None when the meter is unavailable."""
    try:
        return summarize(read_meter(timeout))
    except Exception:
        return None


if __name__ == "__main__":
    r = read_meter()
    if "--raw-keys" in sys.argv:
        print(json.dumps({k: (list(v) if isinstance(v, dict) else v) for k, v in r.items() if k != "accountId"}, default=str))
    print(json.dumps(summarize(r)))
