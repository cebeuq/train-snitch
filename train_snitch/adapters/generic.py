"""Generic adapter: any training run, described declaratively as a list of read-only `sources`.

Every source may live on this machine or on a remote host (`"host": "<name>"` from `hosts`, reached over ssh with
BatchMode). File sources on one host are read together by ONE call of train_snitch/reader.py per cycle (sent over
ssh's stdin for remote hosts), which returns only bytes added since the last call. Command sources run their own
fixed argv (no shell), each on its own minimum interval.

adapter_config:
{
  "hosts":   {"box": {"ssh": "root@203.0.113.7", "port": 22, "options": ["-i", "~/.ssh/run_key"], "ssh_config": null,
                      "python": "python3"}},
  "sources": [
    {"id": "trainer", "type": "log",         "path": "/work/run/train.log", "host": "box", "format": "auto"},
    {"id": "metrics", "type": "jsonl",       "path": "/work/run/metrics.jsonl", "step_key": "step", "loss_key": "loss"},
    {"id": "evals",   "type": "csv",         "path": "/work/run/eval.csv", "step_key": "step", "loss_key": "val_loss", "kind": "val"},
    {"id": "tb",      "type": "tensorboard", "glob": "/work/run/tb/**/events.out.tfevents.*", "loss_tag": "train/loss", "val_tag": "eval/loss"},
    {"id": "wandb",   "type": "wandb_dir",   "path": "/work/run/wandb", "loss_key": "train/loss", "val_loss_key": "eval/loss"},
    {"id": "wandb-api","type": "wandb_api",  "run": "entity/project/run_id", "python": "python3", "loss_key": "train/loss"},
    {"id": "gpu",     "type": "gpu",         "tool": "nvidia-smi", "host": "box"},          (or "rocm-smi"; nvidia "ids": [0, "GPU-..."] = only those)
    {"id": "kernel",  "type": "kernel_log",  "host": "box"},                                 (dmesg / journalctl -k: Xid, OOM)
    {"id": "disk",    "type": "disk",        "path": "/work", "host": "box"},
    {"id": "proc",    "type": "process",     "pattern": "torchrun|train\\.py", "host": "box"},
    {"id": "ckpt",    "type": "checkpoints", "glob": "/work/run/checkpoint-*", "stale_min": 90, "host": "box"},
    {"id": "sync",    "type": "transfer",    "source": {"dir": "/work/run/ckpt", "host": "box"}, "dest": {"dir": "/mnt/nas/run/ckpt"},
                                             "pattern": "**/*", "partial_suffix": ".part"},
    {"id": "vast",    "type": "vast",        "instance_id": 12345678, "credit": true},
    {"id": "runpod",  "type": "runpod",      "pod_id": "abc123", "balance": true},
    {"id": "done",    "type": "done_file",   "path": "/work/run/DONE", "host": "box"},
    {"id": "custom",  "type": "command",     "argv": ["my-status", "--json"], "parse": "json",
                                             "map": {"disk_free_gb": "disk.free_gb", "credit_usd": "billing.credit"}}
  ],
  "planned_rate_usd_h": 15.21, "train_stop_epoch": 1793518083, "protected": ["trainer-a"],
  "every_s": {"files": 60, "gpu": 60, "kernel_log": 120, "process": 60, "vast": 300, "runpod": 300, "wandb_api": 300},
  "ssh_bin": "ssh"
}
Optional per source: "every_s", "host", "first_tail" (lines on first read), "kind" (train | val) for metric files.
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import re
import shlex
import sys

from .base import Adapter
from ..probes import Probe
from .. import parsers, timeutil

READER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reader.py")
FILE_TYPES = {"log", "jsonl", "csv", "tensorboard", "wandb_dir", "disk", "checkpoints", "transfer", "done_file"}
INSTANCE_FIELDS = ("id", "actual_status", "intended_status", "cur_state", "status_msg", "start_date", "duration",
                   "dph_total", "gpu_util", "gpu_temp", "disk_usage", "disk_space", "num_gpus", "gpu_name", "label")
DEFAULT_EVERY = {"files": 60, "gpu": 60, "kernel_log": 120, "process": 60, "vast": 300, "runpod": 300,
                 "wandb_api": 300, "command": 120}

WANDB_API_SCRIPT = r"""
import json, sys
a = json.loads(sys.argv[1])
import wandb
run = wandb.Api(timeout=30).run(a["run"])
keys = [k for k in (a.get("loss_key"), a.get("val_loss_key")) if k]
rows = []
for r in run.scan_history(keys=None, min_step=int(a.get("min_step") or 0), page_size=500):
    rows.append({k: r.get(k) for k in ["_step", "_timestamp"] + keys if k in r})
    if len(rows) >= 2000:
        break
print(json.dumps({"state": run.state, "rows": rows}))
"""


# ---------------------------------------------------------------------------------------------- helpers
def read_new(path: str, state: dict, key: str, first_tail: int = 50, max_bytes: int = 256_000) -> list[str]:
    """New complete lines of a LOCAL file since the last call (byte offset kept in state)."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return []
    off = state.get(key)
    with open(path, "rb") as f:
        if off is None or off > size:
            f.seek(max(0, size - max_bytes))
            lines = f.read().decode("utf-8", "replace").splitlines()[-first_tail:]
            state[key] = size
            return lines
        f.seek(off)
        chunk = f.read(max_bytes)
    end = chunk.rfind(b"\n")
    if end < 0:
        return []
    state[key] = off + end + 1
    return chunk[: end + 1].decode("utf-8", "replace").splitlines()


def parse_jsonl(lines: list[str]) -> list[dict]:
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))  # python json accepts bare NaN / Infinity
        except ValueError:
            continue
    return out


def gpu_rows(lines: list[str]) -> tuple[list, dict]:
    """nvidia-smi csv WITH a timestamp column: timestamp, index, mem, util %, power W, temp, clock."""
    out, now = [], {}
    for ln in lines or []:
        parts = [x.strip() for x in ln.split(",")]
        if len(parts) < 5:
            continue
        try:
            t = timeutil.parse_local(parts[0])
            idx = int(parts[1])
            util = float(parts[3].split()[0])
            pw = float(parts[4].split()[0])
        except (ValueError, IndexError):
            continue
        if t is None:
            continue
        out.append([t, idx, util, pw, ln])
        now[str(idx)] = util
    return out, now


def dig(d, path: str):
    for part in path.split("."):
        if isinstance(d, dict):
            d = d.get(part)
        elif isinstance(d, list) and part.isdigit() and int(part) < len(d):
            d = d[int(part)]
        else:
            return None
    return d


def _num(x):
    return parsers.num(x)


class GenericAdapter(Adapter):
    name = "generic"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.sources = [dict(s) for s in self.acfg.get("sources") or []]
        for i, s in enumerate(self.sources):
            s.setdefault("id", f"{s.get('type', 'src')}{i}")
            s.setdefault("host", "local")
        self.hosts = self.acfg.get("hosts") or {}
        self.every = {**DEFAULT_EVERY, "files": float(cfg.get("cycle_s", 60)), **(self.acfg.get("every_s") or {})}

    # ------------------------------------------------------------------ transport
    def ssh_argv(self, host: str, remote: list[str]) -> list[str]:
        h = self.hosts.get(host)
        if h is None:
            raise ValueError(f"unknown host {host!r} (declare it under adapter_config.hosts)")
        argv = [os.path.expanduser(self.acfg.get("ssh_bin") or "ssh"), "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
        if h.get("ssh_config"):
            argv += ["-F", os.path.expanduser(h["ssh_config"])]
        if h.get("port"):
            argv += ["-p", str(h["port"])]
        argv += [os.path.expanduser(x) for x in h.get("options") or []]
        return argv + [h["ssh"], shlex.join(remote)]

    def on_host(self, host: str, argv: list[str]) -> list[str]:
        return argv if host == "local" else self.ssh_argv(host, argv)

    # ------------------------------------------------------------------ probes
    def _file_ops(self, host: str) -> list[dict]:
        ops = []
        for s in self.sources:
            if s["host"] != host and s["type"] != "transfer":
                continue
            t, i = s["type"], s["id"]
            ft = int(s.get("first_tail", 50))
            if t == "log":
                ops.append({"id": i, "op": "tail", "glob": s.get("path") or s["glob"], "all": bool(s.get("all")), "first_tail": ft})
            elif t == "jsonl":
                ops.append({"id": i, "op": "tail", "glob": s.get("path") or s["glob"], "first_tail": int(s.get("first_tail", 200))})
            elif t == "csv":
                ops.append({"id": i, "op": "csv", "glob": s.get("path") or s["glob"], "first_tail": int(s.get("first_tail", 200))})
            elif t == "tensorboard":
                tags = [x for x in (s.get("loss_tag"), s.get("val_tag")) if x] + list(s.get("tags") or [])
                ops.append({"id": i, "op": "tfevents", "glob": s.get("glob") or os.path.join(s["path"], "**", "events.out.tfevents.*"),
                            "tags": tags})
            elif t == "wandb_dir":
                ops.append({"id": i, "op": "wandb", "path": s["path"], "first_tail": ft})
            elif t == "disk":
                ops.append({"id": i, "op": "disk", "path": s["path"]})
            elif t == "checkpoints":
                ops.append({"id": i, "op": "stat", "glob": s["glob"], "limit": 20})
            elif t == "done_file":
                ops.append({"id": i, "op": "exists", "path": s["path"]})
            elif t == "transfer":
                pat = s.get("pattern", "**/*")
                for side in ("source", "dest"):
                    sd = s[side]
                    if sd.get("host", "local") == host:
                        ops.append({"id": f"{i}:{side}", "op": "stat", "glob": os.path.join(sd["dir"], pat), "limit": 2000})
                        if side == "dest" and s.get("partial_suffix"):
                            ops.append({"id": f"{i}:partial", "op": "stat", "limit": 200,
                                        "glob": os.path.join(sd["dir"], "**", "*" + s["partial_suffix"])})
        return ops

    def probes(self):
        out = []
        hosts = {s["host"] for s in self.sources if s["type"] in FILE_TYPES}
        hosts |= {s[side].get("host", "local") for s in self.sources if s["type"] == "transfer" for side in ("source", "dest")}
        with open(READER) as f:
            reader_src = f.read()
        for host in sorted(hosts):
            ops = self._file_ops(host)
            if not ops:
                continue
            every = min([float(s.get("every_s", self.every["files"])) for s in self.sources
                         if s["type"] in FILE_TYPES and s["host"] == host] or [self.every["files"]])

            def argv_fn(state, host=host, ops=ops):
                req = json.dumps({"ops": ops, "offsets": state.get(f"offsets:{host}") or {}})
                if host == "local":
                    return [sys.executable, "-I", READER, req]
                py = (self.hosts.get(host) or {}).get("python", "python3")
                return self.ssh_argv(host, [py, "-I", "-", req])

            def on_result(res, state, host=host):
                if res.ok and isinstance(res.data, dict):
                    state.setdefault(f"offsets:{host}", {}).update(res.data.get("offsets") or {})

            out.append(Probe(f"files@{host}", argv_fn=argv_fn, stdin=None if host == "local" else reader_src,
                             every_s=every, timeout_s=90, parse="json", on_result=on_result,
                             description=f"read-only file reader on {host}"))
        for s in self.sources:
            p = self._command_probe(s)
            if p:
                out.extend(p if isinstance(p, list) else [p])
        return out

    def _command_probe(self, s):
        t, i, host = s["type"], s["id"], s["host"]
        ev = float(s.get("every_s", self.every.get(t, 120)))
        if t == "gpu":
            if s.get("tool", "nvidia-smi") == "rocm-smi":
                argv, parse = ["rocm-smi", "--showuse", "--showpower", "--json"], "json"
            else:
                argv, parse = ["nvidia-smi", f"--query-gpu={parsers.NVIDIA_QUERY}", "--format=csv,noheader,nounits"], "lines"
                if s.get("ids"):   # only these GPUs (indices or UUIDs): a shared host's other GPUs are not this run's
                    argv[1:1] = ["-i", ",".join(str(x) for x in s["ids"])]
            return Probe(f"cmd:{i}", argv=self.on_host(host, argv), every_s=ev, timeout_s=30, parse=parse)
        if t == "kernel_log":
            sh = s.get("shell") or "dmesg -T 2>/dev/null | tail -n 300 || journalctl -k -n 300 --no-pager 2>/dev/null"
            return Probe(f"cmd:{i}", argv=self.on_host(host, ["sh", "-c", sh]), every_s=ev, timeout_s=30, parse="lines",
                         ok_rc=(0, 1))
        if t == "process":
            return Probe(f"cmd:{i}", argv=self.on_host(host, ["ps", "-eo", "pid,args"]), every_s=ev, timeout_s=30, parse="lines")
        if t == "vast":
            vb = os.path.expanduser(s.get("vastai") or "vastai")
            ps = [Probe(f"cmd:{i}", argv=[vb, "show", "instances", "--raw"], every_s=ev, timeout_s=90, parse="json")]
            if s.get("credit"):
                ps.append(Probe(f"cmd:{i}:credit", argv=[vb, "show", "user", "--raw"], every_s=ev, timeout_s=90, parse="json"))
            return ps
        if t == "runpod":
            rb = os.path.expanduser(s.get("runpodctl") or "runpodctl")
            ps = [Probe(f"cmd:{i}", argv=[rb, "pod", "get", str(s["pod_id"]), "-o", "json"], every_s=ev, timeout_s=60, parse="json")]
            if s.get("balance"):
                ps.append(Probe(f"cmd:{i}:credit", argv=[rb, "user", "-o", "json"], every_s=ev, timeout_s=60, parse="json"))
            return ps
        if t == "wandb_api":
            def argv_fn(state, s=s):
                a = {"run": s["run"], "loss_key": s.get("loss_key", "train/loss"), "val_loss_key": s.get("val_loss_key"),
                     "min_step": (state.get("wandb_step") or {}).get(s["id"], 0)}
                return self.on_host(s["host"], [s.get("python", "python3"), "-I", "-c", WANDB_API_SCRIPT, json.dumps(a)])
            return Probe(f"cmd:{i}", argv_fn=argv_fn, every_s=ev, timeout_s=120, parse="json_last_line")
        if t == "command":
            return Probe(f"cmd:{i}", argv=self.on_host(host, list(s["argv"])), every_s=ev, timeout_s=float(s.get("timeout_s", 60)),
                         parse=s.get("parse", "lines"), ok_rc=tuple(s.get("ok_rc", [0])))
        return None

    # ------------------------------------------------------------------ live collection -> raw snapshot
    def collect(self, results, hist, runner=None):
        import time as _t
        now = _t.time()
        raw: dict = {"now": now, "log_lines": {}, "metric_rows": {}, "source_errors": {}}
        files = {}
        for name, r in results.items():
            if name.startswith("files@"):
                if r.ok:
                    files.update((r.data or {}).get("results") or {})
                else:
                    host = name[6:]
                    for s in self.sources:
                        if s["host"] == host and s["type"] in FILE_TYPES:
                            raw["source_errors"][s["id"]] = (r.stderr_tail or [f"rc {r.rc}"])[-3:]
        ctr = hist.counters.setdefault("generic", {})
        for s in self.sources:
            t, i = s["type"], s["id"]
            res = files.get(i)
            if isinstance(res, dict) and res.get("error"):
                raw["source_errors"][i] = [res["error"]]
                continue
            if t == "log" and res is not None:
                for path, lines in (res.get("files") or {}).items():
                    src = i if not s.get("all") else f"{i}:{os.path.basename(path)}"
                    if lines:
                        raw["log_lines"][src] = lines
                    rows = parsers.log_rows(lines, s.get("format", "auto"), s.get("regex"), ctr.setdefault(f"n:{i}", {"n": 0}))
                    if rows:
                        raw["metric_rows"].setdefault(i, []).extend(rows)
            elif t == "jsonl" and res is not None:
                for path, lines in (res.get("files") or {}).items():
                    rows = [parsers.dict_row(d, s.get("step_key", "step"), s.get("loss_key", "loss"), s.get("kind"),
                                             s.get("kind_key"), s.get("val_loss_key")) for d in parse_jsonl(lines)]
                    raw["metric_rows"].setdefault(i, []).extend(r for r in rows if r)
            elif t == "csv" and res is not None:
                for path, d in (res.get("files") or {}).items():
                    raw["metric_rows"].setdefault(i, []).extend(parsers.csv_rows(
                        d.get("header"), d.get("lines"), step_key=s.get("step_key", "step"), loss_key=s.get("loss_key", "loss"),
                        kind=s.get("kind"), kind_key=s.get("kind_key"), val_loss_key=s.get("val_loss_key")))
            elif t == "tensorboard" and res is not None:
                rows = []
                for p in res.get("points") or []:
                    kind = "train" if p["tag"] == s.get("loss_tag") else "val" if p["tag"] == s.get("val_tag") else None
                    if kind:
                        rows.append({"step": p["step"], "loss": p["value"], "kind": kind, "t": p.get("wall"), "extra": {"tag": p["tag"]}})
                raw["metric_rows"][i] = rows
            elif t == "wandb_dir" and res is not None:
                if res.get("lines"):
                    raw["log_lines"][i] = res["lines"]
                summ = res.get("summary") or {}
                step = summ.get(s.get("step_key", "_step"))
                if step is not None and step != ctr.get(f"wstep:{i}"):
                    ctr[f"wstep:{i}"] = step
                    r = parsers.dict_row(summ, s.get("step_key", "_step"), s.get("loss_key", "loss"), None, None, s.get("val_loss_key"))
                    if r:
                        raw["metric_rows"][i] = [r]
                if res.get("run"):
                    raw.setdefault("run_state", {})[i] = {"run": res["run"], "state": summ.get("_wandb", {}).get("state")
                                                          if isinstance(summ.get("_wandb"), dict) else None}
            elif t == "disk" and res is not None and res.get("free_gb") is not None:
                raw.setdefault("disk", {})[i] = res["free_gb"]
            elif t == "checkpoints" and res is not None:
                it = res.get("items") or []
                raw.setdefault("checkpoints", {})[i] = {"count": res.get("count", 0), "newest": it[0]["path"] if it else None,
                                                        "newest_mtime": it[0]["mtime"] if it else None,
                                                        "stale_min": s.get("stale_min", 90),
                                                        "now": (getattr(results.get(f"files@{s['host']}"), "data", None) or {}).get("now", now)}
            elif t == "done_file" and res is not None:
                raw["done"] = bool(res) or raw.get("done", False)
            elif t == "transfer":
                src, dst, part = files.get(f"{i}:source"), files.get(f"{i}:dest"), files.get(f"{i}:partial")
                if src is not None and dst is not None:
                    def rel(items, d):
                        return {os.path.relpath(x["path"], d): x["size"] for x in items if x.get("size") is not None}
                    raw.setdefault("transfer", {})[i] = {
                        "sources": rel(src.get("items") or [], s["source"]["dir"]),
                        "dest": rel(dst.get("items") or [], s["dest"]["dir"]),
                        "partials": rel((part or {}).get("items") or [], s["dest"]["dir"])}
        for s in self.sources:
            self._collect_command(s, results, raw, hist, runner)
        if self.acfg.get("protected"):
            raw["training_containers"] = self.acfg["protected"]
        for k in ("planned_rate_usd_h", "train_stop_epoch"):
            if self.acfg.get(k) is not None:
                raw[k] = self.acfg[k]
        return raw

    def _collect_command(self, s, results, raw, hist, runner=None):
        t, i = s["type"], s["id"]
        r = results.get(f"cmd:{i}")
        if r is None:
            return
        if not r.ok:
            raw["source_errors"][i] = (r.stderr_tail or [f"rc {r.rc}"])[-3:]
            return
        if t == "gpu":
            rows = parsers.rocm_rows(r.data, r.t) if s.get("tool") == "rocm-smi" else parsers.nvidia_rows(r.data, r.t)
            raw.setdefault("gpu", []).extend(rows)
        elif t == "kernel_log":
            seen = hist.counters.setdefault("generic", {}).setdefault(f"kseen:{i}", [])
            first = not seen
            sset = set(seen)
            new = []
            for ln in r.data or []:
                h = hashlib.sha1(ln.encode()).hexdigest()[:16]
                if h not in sset:
                    sset.add(h)
                    seen.append(h)
                    new.append(ln)
            del seen[:-1000]
            if first:
                new = [ln for ln in new if re.search(r"Xid|Out of memory|oom-kill|NVRM|fallen off the bus|ECC|I/O error", ln)][-20:]
            if new:
                raw["log_lines"][i] = new
        elif t == "process":
            rx = re.compile(s["pattern"])
            hits = [ln.strip() for ln in r.data or [] if rx.search(ln) and "train-snitch" not in ln and "reader.py" not in ln][:5]
            raw.setdefault("processes", {})[i] = {"alive": bool(hits), "count": len(hits), "sample": hits[:2]}
        elif t in ("vast", "runpod"):
            self._billing(s, r, results.get(f"cmd:{i}:credit"), raw)
        elif t == "wandb_api":
            d = r.data or {}
            rows = [parsers.dict_row(x, "_step", s.get("loss_key", "train/loss"), None, None, s.get("val_loss_key")) for x in d.get("rows") or []]
            rows = [x for x in rows if x]
            if rows:
                raw["metric_rows"][i] = rows
                if runner is not None:
                    runner.state.setdefault("wandb_step", {})[i] = max(x["step"] or 0 for x in rows) + 1
            raw.setdefault("run_state", {})[i] = {"run": s["run"], "state": d.get("state")}
        elif t == "command":
            if s.get("parse", "lines") in ("json", "json_last_line"):
                for key, path in (s.get("map") or {}).items():
                    val = dig(r.data, path)
                    if key in ("disk_free_gb", "credit_usd", "planned_rate_usd_h") and _num(val) is not None:
                        raw[key] = _num(val)
                    elif key == "flags" and isinstance(val, dict):
                        raw.setdefault("flags", {}).update(val)
                    elif key == "done":
                        raw["done"] = bool(val) or raw.get("done", False)
            else:
                raw["log_lines"][i] = list(r.data or [])[-200:]

    @staticmethod
    def _billing(s, r, credit, raw):
        if s["type"] == "vast":
            lst = r.data if isinstance(r.data, list) else (r.data or {}).get("instances", [])
            inst = [{k: x.get(k) for k in INSTANCE_FIELDS if k in x} for x in lst if str(x.get("id")) == str(s["instance_id"])]
            if credit is not None and credit.ok and isinstance(credit.data, dict) and _num(credit.data.get("credit")) is not None:
                raw["credit_usd"] = _num(credit.data["credit"])      # only `credit`: the rest of `show user` is private
        else:
            d = r.data if isinstance(r.data, dict) else {}
            if not d:
                inst = []
            else:
                rs = str(d.get("runtimeStatus") or d.get("desiredStatus") or "").lower()
                inst = [{"id": d.get("id"), "actual_status": "running" if rs in ("running", "initializing") else rs,
                         "dph_total": _num(d.get("costPerHr") or d.get("adjustedCostPerHr")), "gpu_name": d.get("gpuDisplayName"),
                         "start_date": None}]
            if credit is not None and credit.ok and isinstance(credit.data, dict):
                bal = credit.data.get("clientBalance", credit.data.get("balance"))
                if _num(bal) is not None:
                    raw["credit_usd"] = _num(bal)
        raw.setdefault("instances", []).extend(inst)
        raw["instances_read"] = True

    # ------------------------------------------------------------------ raw snapshot -> view
    def normalize(self, raw, hist):
        v: dict = {"log_lines": {k: list(x) for k, x in (raw.get("log_lines") or {}).items()}}
        rows = [dict(r) for rs in (raw.get("metric_rows") or {}).values() for r in rs]
        # legacy / replay keys
        tj, vj = self.acfg.get("train_jsonl") or {}, self.acfg.get("val_jsonl") or {}
        for r in raw.get("train_rows") or []:
            rows.append({"step": r.get(tj.get("step_key", "step")), "loss": _num(r.get(tj.get("loss_key", "loss"))), "kind": "train"})
        for r in raw.get("val_rows") or []:
            rows.append({"step": r.get(vj.get("step_key", "step")), "loss": _num(r.get(vj.get("loss_key", "val/loss"))),
                         "kind": r.get(vj.get("kind_key", "kind")) or "val"})
        tr = sorted([r for r in rows if r.get("kind", "train") == "train"], key=lambda r: (r.get("step") is None, r.get("step") or 0))
        va = [dict(r, kind=("val" if r.get("kind") in (None, "val") else r["kind"])) for r in rows if r.get("kind", "train") != "train"]
        if raw.get("metric_rows") is not None or raw.get("train_rows") is not None:
            v["train_rows"] = tr
        if va or raw.get("val_rows") is not None:
            v["val_rows"] = va
        now = raw.get("now")
        g = hist.counters.setdefault("generic", {})
        if now is not None:
            g.setdefault("first_t", now)
            last = hist.last_train_step
            if any(r.get("step") is not None and (last is None or r["step"] > last) for r in tr):
                g["last_row_t"] = now
                g["step"] = max(r["step"] for r in tr if r.get("step") is not None)
            if self._has_metrics():
                procs = raw.get("processes") or {}
                v["progress"] = {"step": g.get("step"), "last_row_t": g.get("last_row_t") or 0, "started_t": g["first_t"],
                                 "alive": any(p.get("alive") for p in procs.values()) if procs else None,
                                 "done": bool(raw.get("done")), "now": now}
        # GPUs
        gl = [ln for k, x in v["log_lines"].items() if k.endswith("gpu.csv") for ln in x]
        if raw.get("gpu"):
            v["gpu_samples"] = [list(x) for x in raw["gpu"]]
            v["gpu_util_now"] = {str(x[1]): x[2] for x in raw["gpu"]}
        elif gl:
            v["gpu_samples"], v["gpu_util_now"] = gpu_rows(gl)
        # disk, money, processes, checkpoints
        disks = list((raw.get("disk") or {}).values()) + ([raw["disk_free_gb"]] if raw.get("disk_free_gb") is not None else [])
        if disks:
            v["disk_free_gb"] = min(float(x) for x in disks)
        for k in ("credit_usd", "planned_rate_usd_h"):
            if raw.get(k) is not None:
                v[k] = float(raw[k])
        if raw.get("instances_read"):
            v["instances"] = raw.get("instances") or []
            if v.get("planned_rate_usd_h") is None:
                rate = next((i.get("dph_total") for i in v["instances"] if i.get("dph_total")), None)
                if rate:
                    v["planned_rate_usd_h"] = float(rate)
        if raw.get("train_stop_epoch"):
            v["deadlines"] = {"train_stop": float(raw["train_stop_epoch"])}
        if raw.get("processes") is not None:
            v["processes"] = raw["processes"]
        if raw.get("checkpoints") is not None:
            v["checkpoints"] = raw["checkpoints"]
        if raw.get("flags") is not None:
            v["flags"] = raw["flags"]
        if raw.get("run_state") is not None:
            v["run_state"] = raw["run_state"]
        # transfers (new shape per source, or the legacy local_files / nas_files pair)
        trs = dict(raw.get("transfer") or {})
        if "nas_files" in raw or "local_files" in raw:
            dest = raw.get("nas_files") or {}
            trs["legacy"] = {"partials": {k: s for k, s in dest.items() if os.path.basename(k).endswith(".part") and s is not None},
                             "sources": raw.get("local_files") or {},
                             "dest": {k: s for k, s in dest.items() if not k.endswith(".part")}}
        if trs:
            m = {"partials": {}, "sources": {}, "dest": {}}
            for name, d in trs.items():
                pre = "" if len(trs) == 1 else f"{name}:"
                for side in m:
                    m[side].update({pre + k: s for k, s in (d.get(side) or {}).items()})
            v["transfers"] = m
        v["protected"] = raw.get("training_containers") or []
        if raw.get("done") is not None:
            v["run_complete"] = {"complete": bool(raw["done"]), "detail": "done marker present"}
        v["source_errors"] = raw.get("source_errors") or {}
        v["_done"] = raw.get("done")
        return v

    def _has_metrics(self):
        return any(s["type"] in ("log", "jsonl", "csv", "tensorboard", "wandb_dir", "wandb_api") for s in self.sources) \
            or self.acfg.get("train_jsonl")

    def done(self, view, hist, t):
        if view.get("_done"):
            return "done marker present"
        for sid, st in (view.get("run_state") or {}).items():
            if str(st.get("state")) in ("finished",):
                return f"W&B run {st.get('run')} finished"
        if view.get("instances") == [] and hist.counters.get("generic", {}).get("instance_seen"):
            return "the rented instance is gone"
        if view.get("instances"):
            hist.counters.setdefault("generic", {})["instance_seen"] = True
        return None

    def summary(self):
        return {"adapter": "generic", "sources": [f"{s['id']}:{s['type']}@{s['host']}" for s in self.sources][:30],
                "run_id": self.acfg.get("run_id") or self.cfg.get("name")}
