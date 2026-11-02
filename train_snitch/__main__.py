"""CLI.

  python3 -m train_snitch run    CONFIG [--once] [--rules-only] [--dry] [--test-alert] [--overrides JSON]
                                                                          watch (finite; exits at run end/deadline)
  python3 -m train_snitch check  CONFIG [--overrides JSON]                validate a config + show the judge model chain
  python3 -m train_snitch replay CONFIG INCIDENT_DIR                      replay a recorded incident, print alerts
  python3 -m train_snitch tail   OUT_DIR [--min warn] [--from-start]      follow alerts (for an agent's Monitor tool)
  python3 -m train_snitch meter                                           Codex usage meter + credits (no model call)
  python3 -m train_snitch eval-judge CONFIG [--once] [--dry] [--no-act]    vision-model judge of eval image sets
  python3 -m train_snitch mcp                                             MCP stdio server (needs the [mcp] extra)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from . import config as cfgmod
from .engine import SEV


def cmd_run(a):
    from .watcher import Watcher
    cfg = cfgmod.load(a.config, _overrides(a))
    if a.out:
        cfg["out_dir"] = os.path.abspath(a.out)
    w = Watcher(cfg, rules_only=a.rules_only, dry=a.dry)
    print(w.run(once=a.once, test_alert=a.test_alert))


def _overrides(a):
    if not getattr(a, "overrides", None):
        return None
    try:
        ov = json.loads(a.overrides)
    except ValueError as e:
        raise SystemExit(f"--overrides must be a JSON object: {e}")
    if not isinstance(ov, dict):
        raise SystemExit("--overrides must be a JSON object")
    return ov


def cmd_check(a):
    from . import llm
    cfg = cfgmod.load(a.config, _overrides(a))
    j = cfg["judge"]
    errs, warns = llm.validate(j) if j.get("enabled") else ([], [])
    print(json.dumps({"ok": not errs, "errors": errs, "warnings": warns, "name": cfg.get("name"), "out_dir": cfg["out_dir"],
                      "judge": llm.describe(j) if j.get("enabled") else "off (rules only)",
                      "meter": llm.meter_kind(j) if j.get("enabled") else None}, indent=1))
    if errs:
        sys.exit(1)


def cmd_mcp(_a):
    from .mcp_server import main as mcp_main
    mcp_main()


def cmd_replay(a):
    from .replay import replay
    cfg = cfgmod.load(a.config, {"judge": {"enabled": False}})
    for i, c in enumerate(replay(a.incident, cfg)):
        for r in c["records"]:
            print(f"[{i}] {c['t_local']}: {r['severity']:8} {r['status']:9} {r['rule']}: {r['what']}")


def fmt(r):
    ev = (" | " + r["evidence"][0][:160]) if r.get("evidence") else ""
    act = (" | action: " + r["action"][:200]) if r.get("action") else ""
    return f"{r['severity'].upper()} {r['status']} {r.get('time', '')}: {r['what']}{ev}{act}"


def cmd_tail(a):
    """Print one line per new alert at/above --min; exit when the watcher writes its exit record.
    Also prints one line if the watcher goes silent (status.json not updated for --stale-min)."""
    path = os.path.join(a.out_dir, "alerts.jsonl")
    status = os.path.join(a.out_dir, "status.json")
    while not os.path.exists(path):
        time.sleep(2)
    with open(path) as f:
        _follow(f, a, status)


def _follow(f, a, status):
    if not a.from_start:
        f.seek(0, 2)
    warned_stale = False
    seen_alive = False
    buf = ""
    while True:
        chunk = f.readline()
        if chunk:
            buf += chunk
            if not buf.endswith("\n"):
                continue
            line, buf = buf, ""
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("rule") == "watcher_exit":
                pos = f.tell()
                if not f.readline():          # the newest record: the watcher is gone, so is this tail
                    print(fmt(r), flush=True)
                    return
                f.seek(pos)                   # an older run's exit (watcher was restarted): keep following
                continue
            judge_note = r.get("source") == "judge" and r.get("about")      # "the judge thinks alert X is a false alarm"
            judge_note = judge_note or r.get("source") == "eval_judge"     # every eval-judge verdict / action
            if judge_note or r.get("test") or SEV.get(r.get("severity"), 0) >= SEV[a.min] and (a.all or r.get("status") != "ongoing" or r["severity"] == "critical") \
                    and not (r.get("quiet") and not a.all):
                print(fmt(r), flush=True)
            continue
        alive = _watcher_alive(a.out_dir)
        if alive:
            seen_alive = True
        elif seen_alive or (alive is False and not _exited(status)):
            # the watcher died without writing its exit record (killed -9, power loss): this tail ends too
            print(f"WATCHER GONE: no watcher process for {a.out_dir} and no exit record (killed or crashed); "
                  f"restart it if the run is still live", flush=True)
            return
        try:
            age = time.time() - os.path.getmtime(status)
        except OSError:
            age = 0
        if age > a.stale_min * 60 and not warned_stale:
            print(f"WATCHER SILENT: status.json not updated for {int(age // 60)} min (watcher crashed or hung?)", flush=True)
            warned_stale = True
        elif age <= a.stale_min * 60:
            warned_stale = False
        time.sleep(1)


def _watcher_alive(out_dir):
    """True: watcher.pid names a live process. False: pid file names a dead process. None: no pid file yet."""
    try:
        with open(os.path.join(out_dir, "watcher.pid")) as f:
            pid = int(f.read().strip())
    except (OSError, ValueError):
        return None
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _exited(status):
    try:
        with open(status) as f:
            return bool(json.load(f).get("exited"))
    except (OSError, ValueError):
        return False


def cmd_eval_judge(a):
    from .evaljudge import EvalJudge, load_config
    over = {}
    if a.steps:
        over["source"] = {"steps": [int(x) for x in a.steps.split(",")]}
    cfg = load_config(a.config, over)
    if a.out:
        cfg["out_dir"] = os.path.abspath(a.out)
    j = EvalJudge(cfg, dry=a.dry, no_act=a.no_act, log=lambda m: print(m, flush=True))
    print(j.run(once=a.once))


def cmd_meter(_a):
    from .meter import read_summary
    print(json.dumps(read_summary()))


def main(argv=None):
    p = argparse.ArgumentParser(prog="train-snitch")
    sp = p.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("config")
    r.add_argument("--once", action="store_true", help="one cycle, then exit")
    r.add_argument("--rules-only", action="store_true", help="never call the LLM judge")
    r.add_argument("--dry", action="store_true", help="no desktop notification / webhook / command hook")
    r.add_argument("--out", help="override out_dir")
    r.add_argument("--test-alert", action="store_true", help="send one info TEST alert through every channel at start")
    r.add_argument("--overrides", help="JSON object merged over the config (e.g. a judge model for this run)")
    r.set_defaults(fn=cmd_run)
    r = sp.add_parser("check", help="validate a config and print the judge model chain (no model call)")
    r.add_argument("config")
    r.add_argument("--overrides")
    r.set_defaults(fn=cmd_check)
    r = sp.add_parser("mcp", help="run the MCP stdio server for agent harnesses (pip install 'train-snitch[mcp]')")
    r.set_defaults(fn=cmd_mcp)
    r = sp.add_parser("replay")
    r.add_argument("config")
    r.add_argument("incident")
    r.set_defaults(fn=cmd_replay)
    r = sp.add_parser("tail")
    r.add_argument("out_dir")
    r.add_argument("--min", default="warn", choices=list(SEV))
    r.add_argument("--all", action="store_true", help="also ongoing reminders and quiet judge notes")
    r.add_argument("--from-start", action="store_true")
    r.add_argument("--stale-min", type=float, default=15)
    r.set_defaults(fn=cmd_tail)
    r = sp.add_parser("eval-judge", help="judge each new eval image set with a vision model (finite)")
    r.add_argument("config")
    r.add_argument("--once", action="store_true", help="one poll, then exit")
    r.add_argument("--dry", action="store_true", help="build sheets, no model call, no action")
    r.add_argument("--no-act", action="store_true", help="never run the stop command (recommend only)")
    r.add_argument("--out", help="override out_dir")
    r.add_argument("--steps", help="only these steps, comma-separated (replay)")
    r.set_defaults(fn=cmd_eval_judge)
    r = sp.add_parser("meter")
    r.set_defaults(fn=cmd_meter)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
