"""Config: one JSON file merged over DEFAULTS. Relative paths in the file resolve against the file's folder."""
from __future__ import annotations

import copy
import importlib.util
import json
import os

DEFAULTS: dict = {
    "name": "run",
    "timezone": None,                    # e.g. "America/Los_Angeles"; None = this machine's zone
    "out_dir": "./out",                  # alerts.jsonl, latest.txt, status.json, state.json, watcher.log
    "adapter": "generic",                # "generic" or "path/to/adapter.py:ClassName"
    "adapter_config": {},
    "cycle_s": 60,
    "lifecycle": {"max_hours": 12, "deadline_epoch": None, "deadline_local": None, "grace_after_done_s": 0,
                  "max_cycle_errors": 10},     # failed cycles in a row before the watcher gives up (with an exit alert)
    "engine": {"clear_cycles": 1, "event_ttl_s": 2700, "remind_critical_s": 1800, "remind_warn_s": 7200},
    "vars": {},                          # extra {placeholders} for rule action texts
    "flags": {},
    "rules": {},
    # Generic training-failure signatures (events). Adapters/configs add their own after these.
    "log_patterns": [
        {"id": "log:oom", "severity": "critical", "pattern": r"CUDA out of memory|OutOfMemoryError",
         "what": "CUDA out of memory in {source}",
         "action": "Check what ran next to training at that moment (gate / sample renders, val pass) and the memory headroom; "
                   "auto-resume (if any) restarts from the newest resume save."},
        {"id": "log:xid", "severity": "critical", "pattern": r"NVRM: Xid \((?P<pci>[^)]*)\): (?P<xid>\d+)",
         "what": "GPU Xid {xid} on {pci}", "action": "Hardware/driver fault: the GPU may be gone; check nvidia-smi and the trainer. "
                                                     "A host with repeated Xid 79/48/63 needs replacing."},
        {"id": "log:nccl_timeout", "severity": "critical",
         "pattern": r"Watchdog caught collective operation timeout|NCCL.*[Tt]imed? ?out|ProcessGroupNCCL.*Timeout",
         "what": "NCCL collective timeout in {source}: a rank is hung or dead", "judge": True,
         "action": "The job usually aborts next; check which rank stopped (GPU util, per-rank logs) and resume from the last save."},
        {"id": "log:nccl_error", "severity": "warn",
         "pattern": r"NCCL (?:WARN|ERROR)\b|ncclInternalError|ncclSystemError|ncclUnhandledCudaError|ncclRemoteError",
         "ignore": [r"Warning: An unbatched P2P op"], "what": "NCCL error in {source}", "judge": True,
         "action": "A rank may hang next: watch for stalled rows."},
        {"id": "log:oom_killer", "severity": "critical", "pattern": r"Out of memory: Killed process|oom-kill|oom_reaper",
         "what": "Host OOM killer fired ({source})", "action": "Host RAM exhausted: lower dataloader workers / caching."},
        {"id": "log:no_space", "severity": "critical", "pattern": r"No space left on device|Disk quota exceeded",
         "what": "Disk full ({source})", "action": "Free disk now; the next save will fail."},
        {"id": "log:worker_failed", "severity": "critical",
         "pattern": r"ChildFailedError|torch\.distributed\.elastic\.multiprocessing\.errors|exitcode\s*:\s*-?\d+.*(?:SIGKILL|SIGSEGV|SIGABRT)|"
                    r"Signal \d+ \((?:SIGKILL|SIGSEGV|SIGABRT|SIGBUS)\) received",
         "what": "A distributed worker failed ({source}): {line}",
         "action": "Read the first traceback above this line (the root cause is on the first failing rank); resume from the last save."},
        {"id": "log:cuda_error", "severity": "critical",
         "pattern": r"CUDA error: |CUBLAS_STATUS_\w+|CUDNN_STATUS_\w+|cudaErrorIllegalAddress|illegal memory access|"
                    r"device-side assert triggered|HIP error: |hipError\w+",
         "what": "CUDA/HIP runtime error in {source}", "action": "Usually fatal for the process; check for an Xid in the kernel log (hardware) vs a code bug."},
        {"id": "log:gpu_lost", "severity": "critical",
         "pattern": r"GPU has fallen off the bus|uncorrectable ECC|ECC error|Unable to determine the device handle",
         "ignore": [r"NVRM:"], "what": "GPU hardware fault ({source}): {line}", "action": "The GPU may be gone: check nvidia-smi; this host likely needs replacing."},
        {"id": "log:segfault", "severity": "critical", "pattern": r"Segmentation fault|core dumped|Bus error|Fatal Python error",
         "what": "Process crashed hard in {source}: {line}", "action": "Read the log above; resume from the last save."},
        {"id": "log:traceback", "severity": "warn", "pattern": r"^Traceback \(most recent call last\)", "judge": True,
         "what": "Python traceback in {source}", "action": "Check whether the trainer survived it (process alive, rows still arriving)."},
        {"id": "log:grad_overflow", "severity": "warn", "min_count": 5,
         "pattern": r"OVERFLOW! Rank \d+ Skipping step|Gradient overflow\.\s+Skipping step|Attempted loss scale: \S+, reducing to",
         "what": "Repeated fp16 gradient overflows in {source}: steps are being skipped", "judge": True,
         "action": "A few at the start are normal (loss-scale search); many in a row mean divergence: compare loss / grad norm."},
        {"id": "log:grad_nonfinite", "severity": "warn", "pattern": r"grad_norm['\"]?\s*[:=]\s*(?:nan|inf)\b",
         "what": "Gradient norm is NaN/inf in {source}", "action": "Precursor of a NaN loss: check lr, data and loss scale."},
    ],
    "evidence_patterns": [r"Error|error:|FAILED|Traceback|Resource temporarily unavailable|fork:|Killed|Exception|rc=[1-9]"],
    "benign_patterns": [r"Warning: An unbatched P2P op", r"libpng warning", r"Welcome to vast\.ai", r"destroy_process_group"],
    "judge": {
        "enabled": True,
        "model": "openai-codex/gpt-6-luna",   # provider/id as Pi knows it (e.g. anthropic/<haiku id> with "meter": "none")
        "every_s": 60,                   # cadence: one judgment per this many seconds (never more often than cycle_s)
        "skip_unchanged": False,         # true: skip a tick when nothing changed (still at least every max_quiet_min)
        "max_quiet_min": 15,
        "agent_on_anomaly": True,        # new warn+ rule alert -> Pi AgentSession with read-only tools
        "max_turns": 3,                  # agent calls: model turns per judgment (each counts as one call)
        "credit_cap": 150,               # Codex credits per watcher run; above it -> rules-only + one alert
        "meter_every_calls": 15,         # read the Codex meter (no model call) every N judgments
        "max_consecutive_errors": 3,     # failed calls in a row -> rules-only + one alert
        "max_calls": 2000,               # hard backstop on model turns per run
        "snitch_repeat_min": 30,         # same judge topic inside this window is not re-alerted
        "probe_gap_min": 15,             # adapter judge probes (e.g. one box log) at most this often
        "reasoning_effort": "none",
        "max_tokens": 500,
        "max_prompt_chars": 9000,
        "timeout_s": 90,
        "node": None,                    # node binary; None = from PATH
        "pi_pkg": None,                  # Pi package dir; None = $PI_PKG or `npm root -g`
        "bridge": None,                  # None = <repo>/judge/pi_judge.mjs
        "meter": "codex",                # codex | none
    },
    "channels": {
        "jsonl": True, "latest": True,
        "notify": {"enabled": True, "min_severity": "warn", "title": "train-snitch"},
        "webhook": {"url": None, "min_severity": "warn", "timeout_s": 10},
        "command": {"argv": None, "min_severity": "warn", "timeout_s": 20},
    },
}


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if k == "log_patterns_extra":
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load(path: str | None, overrides: dict | None = None) -> dict:
    user = {}
    base = os.getcwd()
    if path:
        user = _read_chain(path)
        base = os.path.dirname(os.path.abspath(path))
    cfg = deep_merge(DEFAULTS, user)
    cfg["log_patterns"] = list(cfg["log_patterns"]) + list(user.get("log_patterns_extra") or [])
    cfg = deep_merge(cfg, overrides or {})
    cfg["_base"] = base
    cfg["out_dir"] = resolve(cfg, cfg["out_dir"])
    return cfg


def _read_chain(path: str, depth: int = 0) -> dict:
    """A config may say {"extends": "other.json", ...}: the other file is loaded first (path relative to this file)
    and this file's keys are merged over it (log_patterns_extra lists are concatenated). Paths inside the parent
    resolve against the child's folder, so keep both files in one folder."""
    with open(path) as f:
        user = json.load(f)
    parent = user.pop("extends", None)
    if not parent or depth > 5:
        return user
    pp = os.path.join(os.path.dirname(os.path.abspath(path)), os.path.expanduser(parent))
    base = _read_chain(pp, depth + 1)
    extra = list(base.get("log_patterns_extra") or []) + list(user.get("log_patterns_extra") or [])
    out = deep_merge(base, user)
    if extra:
        out["log_patterns_extra"] = extra
    return out


def resolve(cfg: dict, p: str | None) -> str | None:
    if not p:
        return p
    p = os.path.expanduser(p)
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(cfg.get("_base", os.getcwd()), p))


def load_adapter(cfg: dict):
    spec = cfg.get("adapter") or "generic"
    if spec in ("generic", "generic-files"):
        from .adapters.generic import GenericAdapter
        return GenericAdapter(cfg)
    path, _, cls = spec.partition(":")
    path = resolve(cfg, path)
    import sys
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo not in sys.path:
        sys.path.insert(0, repo)
    m = importlib.util.spec_from_file_location(f"ts_adapter_{abs(hash(path))}", path)
    mod = importlib.util.module_from_spec(m)
    m.loader.exec_module(mod)
    return getattr(mod, cls or "Adapter")(cfg)
