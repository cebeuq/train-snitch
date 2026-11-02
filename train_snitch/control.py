"""Watcher control for agent harnesses: start / stop / list / status / alerts. Stdlib only.

`mcp_server.py` exposes these functions as MCP tools; they are plain functions so they work (and are tested)
without the MCP SDK. Each one is a thin wrapper over the CLI and the files a watcher writes in its out_dir.

Safety boundary (read-only by design):
- A config may define probe commands and a command hook, so `start_watch` only accepts config files under
  TRAIN_SNITCH_CONFIG_DIRS, and inline `overrides` are limited to a small allowlist (no sources, hosts, channels,
  adapters, commands, endpoints or binaries).
- Every out_dir must be under TRAIN_SNITCH_OUT_DIRS (default: the config dirs).
- `stop_watch` only sends SIGTERM to the pid in <out_dir>/watcher.pid, and only when that process's command line
  is a train-snitch watcher. It never touches the training run.
- No shell, no arbitrary commands, no file writes except the watcher's own out_dir (made by the watcher).
Both dir variables are path lists (":"-separated, ";" on Windows). Unset: the server's working directory, unless
that is the home directory or "/" (then nothing is allowed until they are set).
"""
from __future__ import annotations

import json
import os
import re
import signal
import stat
import subprocess
import sys
import time

from . import config as cfgmod
from . import llm
from .engine import SEV

ENV_CONFIG_DIRS = "TRAIN_SNITCH_CONFIG_DIRS"
ENV_OUT_DIRS = "TRAIN_SNITCH_OUT_DIRS"
ENV_FILE = "TRAIN_SNITCH_ENV_FILE"
PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# python -I -c BOOT PKG_ROOT run ...: works from a checkout and from an installed package alike
BOOT = "import sys; sys.path.insert(0, sys.argv.pop(1)); from train_snitch.__main__ import main; main()"
WATCHER_CMD = re.compile(r"train[-_]snitch")

# inline overrides an agent may pass to start_watch / check_config (anything else must live in a config file)
SAFE_TOP = {"name", "cycle_s", "timezone", "lifecycle", "rules", "judge"}
SAFE_LIFECYCLE = {"max_hours", "deadline_epoch", "deadline_local"}
SAFE_JUDGE = {"enabled", "model", "fallbacks", "fallback_on", "fallback_cooldown_min", "every_s", "reasoning",
              "temperature", "max_tokens", "max_calls", "credit_cap", "cost_cap_usd", "skip_unchanged",
              "max_quiet_min", "agent_on_anomaly", "max_turns", "timeout_s", "structured_output"}

_CHILDREN: dict[int, subprocess.Popen] = {}   # watchers started by this process (reaped on every call)


class ControlError(Exception):
    """A refused or failed request; the message is safe to show to the agent."""


# ---------------------------------------------------------------- allowlisted roots
def _roots(var: str, fallback: list[str] | None = None) -> list[str]:
    raw = os.environ.get(var, "")
    parts = [p for p in raw.split(os.pathsep) if p.strip()]
    if parts:
        return [os.path.realpath(os.path.expanduser(p.strip())) for p in parts]
    if fallback is not None:
        return fallback
    cwd = os.path.realpath(os.getcwd())
    if cwd in (os.path.realpath(os.path.expanduser("~")), os.path.realpath(os.sep)):
        return []
    return [cwd]


def config_roots() -> list[str]:
    return _roots(ENV_CONFIG_DIRS)


def out_roots() -> list[str]:
    return _roots(ENV_OUT_DIRS, config_roots())


def _under(path: str, roots: list[str]) -> bool:
    p = os.path.realpath(path)
    for r in roots:
        try:
            if os.path.commonpath([p, r]) == r:
                return True
        except ValueError:
            continue
    return False


def _need_roots(roots: list[str], var: str) -> None:
    if not roots:
        raise ControlError(f"no allowed folders: set {var} (a {os.pathsep!r}-separated list of absolute paths) in this "
                           "MCP server's environment; see docs/HARNESSES.md")


# ---------------------------------------------------------------- config
def check_overrides(ov) -> list[str]:
    if ov is None:
        return []
    if not isinstance(ov, dict):
        return ["overrides must be an object"]
    errs = [f"overrides.{k} is not allowed inline (put it in the config file)" for k in ov if k not in SAFE_TOP]
    for k in set(ov.get("lifecycle") or {}) - SAFE_LIFECYCLE:
        errs.append(f"overrides.lifecycle.{k} is not allowed inline")
    j = ov.get("judge")
    if j is not None:
        if not isinstance(j, dict):
            errs.append("overrides.judge must be an object")
        else:
            errs += [f"overrides.judge.{k} is not allowed inline (endpoints, keys and binaries go in the config file)"
                     for k in j if k not in SAFE_JUDGE]
            if any(not isinstance(f, str) for f in j.get("fallbacks") or []):
                errs.append("overrides.judge.fallbacks: only \"provider/id\" strings inline")
    r = ov.get("rules")
    if r is not None and (not isinstance(r, dict) or any(not isinstance(v, dict) for v in r.values())):
        errs.append("overrides.rules must map rule names to objects (thresholds or {\"enabled\": false})")
    return errs


def load_config(config: str, overrides: dict | None = None) -> dict:
    roots = config_roots()
    _need_roots(roots, ENV_CONFIG_DIRS)
    if not isinstance(config, str) or not config.endswith(".json"):
        raise ControlError("config must be the path of a .json run config")
    path = os.path.realpath(os.path.expanduser(config))
    if not _under(path, roots):
        raise ControlError(f"config {path} is outside the allowed config folders ({ENV_CONFIG_DIRS}: {', '.join(roots)})")
    if not os.path.isfile(path):
        raise ControlError(f"config not found: {path}")
    errs = check_overrides(overrides)
    if errs:
        raise ControlError("; ".join(errs))
    try:
        cfg = cfgmod.load(path, overrides or None)
    except (OSError, ValueError) as e:
        raise ControlError(f"cannot read config {path}: {e}") from None
    oroots = out_roots()
    if not _under(cfg["out_dir"], oroots):
        raise ControlError(f"the config's out_dir {cfg['out_dir']} is outside the allowed output folders "
                           f"({ENV_OUT_DIRS}: {', '.join(oroots) or 'none'})")
    cfg["_path"] = path
    return cfg


def check_config(config: str, overrides: dict | None = None) -> dict:
    """Validate a run config (and inline overrides) without running anything: judge model chain, out_dir."""
    cfg = load_config(config, overrides)
    j = cfg["judge"]
    errs, warns = llm.validate(j) if j.get("enabled") else ([], [])
    ac = cfg.get("adapter_config") or {}
    return {"ok": not errs, "errors": errs, "warnings": warns, "config": cfg["_path"], "name": cfg.get("name"),
            "out_dir": cfg["out_dir"], "adapter": cfg.get("adapter"), "sources": len(ac.get("sources") or []),
            "judge": llm.describe(j) if j.get("enabled") else "off (rules only)",
            "meter": llm.meter_kind(j) if j.get("enabled") else None, "cost_cap_usd": j.get("cost_cap_usd"),
            "credit_cap": j.get("credit_cap")}


# ---------------------------------------------------------------- env for the detached watcher
def _env_file_vars(path: str) -> dict[str, str]:
    """KEY=VALUE lines from a private dotenv file (for harnesses that pass the server no env). Values are never
    returned to the caller or logged."""
    p = os.path.realpath(os.path.expanduser(path))
    try:
        st = os.stat(p)
    except OSError:
        raise ControlError(f"{ENV_FILE} points to a missing file") from None
    if not stat.S_ISREG(st.st_mode):
        raise ControlError(f"{ENV_FILE} must be a regular file")
    if hasattr(os, "getuid") and (st.st_uid != os.getuid() or st.st_mode & 0o077):
        raise ControlError(f"{ENV_FILE} must be owned by you and private (chmod 600)")
    out = {}
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line = line[7:] if line.startswith("export ") else line
            k, sep, v = line.partition("=")
            k, v = k.strip(), v.strip()
            if sep and llm.ENV_NAME.match(k):
                if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
                    v = v[1:-1]
                out[k] = v
    return out


def child_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("HOME", os.path.expanduser("~"))   # some harnesses pass only PATH; Pi needs HOME for its login
    ef = env.get(ENV_FILE)
    if ef:
        for k, v in _env_file_vars(ef).items():
            if not env.get(k):
                env[k] = v
    return env


# ---------------------------------------------------------------- processes
def _reap() -> None:
    for pid, p in list(_CHILDREN.items()):
        if p.poll() is not None:
            del _CHILDREN[pid]


def _read_pid(out_dir: str):
    try:
        with open(os.path.join(out_dir, "watcher.pid")) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _alive(pid) -> bool:
    if not pid:
        return False
    if pid in _CHILDREN and _CHILDREN[pid].poll() is not None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        pass
    try:
        return subprocess.run(["ps", "-o", "args=", "-p", str(pid)], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _is_watcher(pid: int) -> bool:
    c = _cmdline(pid)
    return bool(WATCHER_CMD.search(c)) and bool(re.search(r"(?:^|\s)run(?:\s|$)", c))


def _read_json(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------- runs (out_dirs)
def _is_out_dir(d: str) -> bool:
    return any(os.path.exists(os.path.join(d, n)) for n in ("status.json", "watcher.pid", "alerts.jsonl"))


def _summary(d: str) -> dict:
    st = _read_json(os.path.join(d, "status.json")) or {}
    pid = _read_pid(d)
    alive = _alive(pid) and _is_watcher(pid)
    active = st.get("active") or []
    worst = max((a.get("severity", "info") for a in active), key=lambda s: SEV.get(s, 0), default=None)
    return {"run": st.get("name") or os.path.basename(d), "out_dir": d, "alive": alive, "pid": pid if alive else None,
            "updated": st.get("updated"), "cycle": st.get("cycle"), "deadline": st.get("deadline"),
            "judge_mode": st.get("judge_mode"), "active_alerts": len(active), "worst_active": worst,
            "exited": st.get("exited")}


def list_watches() -> dict:
    """Every watcher out_dir found under the allowed output folders (up to 3 levels deep)."""
    _reap()
    roots = out_roots()
    _need_roots(roots, ENV_OUT_DIRS)
    found, seen = [], 0
    for r in roots:
        base = r.rstrip(os.sep).count(os.sep)
        for d, dirs, _files in os.walk(r):
            seen += 1
            dirs[:] = sorted(x for x in dirs if not x.startswith(".") and x not in ("node_modules", "__pycache__"))
            if _is_out_dir(d):
                found.append(_summary(d))
                dirs[:] = []
            if d.count(os.sep) - base >= 3 or seen > 5000:
                dirs[:] = []
    found.sort(key=lambda x: (not x["alive"], x["run"]))
    return {"watches": found, "roots": roots}


def resolve_run(run: str) -> str:
    """A run is its out_dir path, or a run name / folder name from list_watches (must be unique)."""
    if not isinstance(run, str) or not run.strip():
        raise ControlError("run must be an out_dir path or a run name from list_watches")
    roots = out_roots()
    _need_roots(roots, ENV_OUT_DIRS)
    if os.sep in run or run.startswith("~"):
        d = os.path.realpath(os.path.expanduser(run))
        if not _under(d, roots):
            raise ControlError(f"{d} is outside the allowed output folders ({ENV_OUT_DIRS})")
        if not os.path.isdir(d):
            raise ControlError(f"no such out_dir: {d}")
        return d
    hits = [w["out_dir"] for w in list_watches()["watches"] if run in (w["run"], os.path.basename(w["out_dir"]))]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise ControlError(f"no watcher named {run!r} under {', '.join(roots)}; call list_watches")
    raise ControlError(f"{run!r} is ambiguous ({', '.join(hits)}); pass the out_dir path")


def status(run: str) -> dict:
    d = resolve_run(run)
    st = _read_json(os.path.join(d, "status.json")) or {}
    return {**_summary(d), "status": st}


def latest(run: str) -> dict:
    d = resolve_run(run)
    try:
        with open(os.path.join(d, "latest.txt")) as f:
            text = f.read(20000)
    except OSError:
        text = ""
    return {"run": d, "latest": text or "(no warn+ alert yet)"}


def _tail_lines(path: str, max_bytes: int = 512 * 1024) -> list[str]:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            data = f.read()
    except OSError:
        return []
    lines = data.decode("utf-8", "replace").splitlines()
    return lines[1:] if size > max_bytes else lines


def tail_alerts(run: str, n: int = 20, min_severity: str = "warn", since: float | None = None,
                include_ongoing: bool = False) -> dict:
    """The newest alerts, filtered like `train-snitch tail`: warn+ (or min_severity), judge false-alarm notes,
    eval-judge records, TEST records and the watcher's exit record. Pass `next_since` back as `since` to follow."""
    d = resolve_run(run)
    if min_severity not in SEV:
        raise ControlError(f"min_severity must be one of {', '.join(SEV)}")
    n = max(1, min(int(n or 20), 200))
    out = []
    for line in _tail_lines(os.path.join(d, "alerts.jsonl")):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if since is not None and float(r.get("t") or 0) <= float(since):
            continue
        special = (r.get("source") == "judge" and r.get("about")) or r.get("source") == "eval_judge" or r.get("test") \
            or r.get("rule") in ("watcher_exit", "watcher_start", "judge_off")
        if not special:
            if SEV.get(r.get("severity"), 0) < SEV[min_severity]:
                continue
            if r.get("status") == "ongoing" and r.get("severity") != "critical" and not include_ongoing:
                continue
        out.append(r)
    out = out[-n:]
    nxt = max([float(r.get("t") or 0) for r in out] + [float(since or 0)])
    s = _summary(d)
    return {"run": d, "alive": s["alive"], "alerts": out, "next_since": nxt or None}


# ---------------------------------------------------------------- start / stop
def start_watch(config: str, overrides: dict | None = None, rules_only: bool = False, dry: bool = False,
                once: bool = False, test_alert: bool = False, wait_s: float = 8.0) -> dict:
    """Start a detached watcher for an allowed config. Finite: it exits by itself at run end or its deadline."""
    _reap()
    cfg = load_config(config, overrides)
    warns = []
    if cfg["judge"].get("enabled") and not rules_only:
        errs, warns = llm.validate(cfg["judge"])
        if errs:
            raise ControlError("invalid judge config: " + "; ".join(errs))
    out = cfg["out_dir"]
    pid = _read_pid(out)
    if _alive(pid) and _is_watcher(pid):
        raise ControlError(f"a watcher (pid {pid}) already writes to {out}; stop_watch it first or use another out_dir")
    os.makedirs(out, exist_ok=True)
    argv = [sys.executable, "-I", "-c", BOOT, PKG_ROOT, "run", cfg["_path"]]
    argv += [f for f, on in (("--rules-only", rules_only), ("--dry", dry), ("--once", once), ("--test-alert", test_alert)) if on]
    if overrides:
        argv += ["--overrides", json.dumps(overrides, sort_keys=True)]
    env = child_env()
    with open(os.path.join(out, "watcher.out"), "ab") as log:
        p = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env,
                             cwd=os.path.dirname(cfg["_path"]), start_new_session=True, close_fds=True)
    _CHILDREN[p.pid] = p
    t_end = time.time() + wait_s
    while time.time() < t_end:
        if _read_pid(out) == p.pid or p.poll() is not None:
            break
        time.sleep(0.1)
    rc = p.poll()
    res = {"run": cfg.get("name"), "out_dir": out, "pid": p.pid, "warnings": warns,
           "judge": "off (rules only)" if rules_only or not cfg["judge"].get("enabled") else llm.describe(cfg["judge"]),
           "follow": f"tail_alerts(run={out!r}, since=<next_since>) or `train-snitch tail {out}`"}
    if rc is None:
        return {"started": True, **res}
    tail = _tail_lines(os.path.join(out, "watcher.out"), 4000)[-12:]
    return {"started": rc == 0 and once, "exited": True, "returncode": rc, "output_tail": tail, **res}


def stop_watch(run: str, wait_s: float = 15.0) -> dict:
    """SIGTERM the watcher of one out_dir (never the training run). The watcher writes its exit record and exits."""
    _reap()
    d = resolve_run(run)
    pid = _read_pid(d)
    if not _alive(pid):
        return {"run": d, "stopped": False, "reason": "no live watcher for this out_dir"}
    if not _is_watcher(pid):
        raise ControlError(f"pid {pid} in {d}/watcher.pid is not a train-snitch watcher; refusing to signal it")
    os.kill(pid, signal.SIGTERM)
    t_end = time.time() + wait_s
    while time.time() < t_end and _alive(pid):
        if pid in _CHILDREN:
            try:
                _CHILDREN[pid].wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                pass
        else:
            time.sleep(0.2)
    _reap()
    gone = not _alive(pid)
    st = _read_json(os.path.join(d, "status.json")) or {}
    return {"run": d, "stopped": gone, "pid": pid, "exited": st.get("exited"),
            **({} if gone else {"reason": f"still running after {wait_s:g} s (it finishes its current cycle first)"})}


def meter() -> dict:
    """Codex usage meter + credit balance (no model call). None fields when the Codex CLI is not logged in."""
    from .meter import read_summary
    return {"meter": read_summary()}
