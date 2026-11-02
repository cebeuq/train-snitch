"""Eval judge: a vision model judges each new set of eval images (sample renders) a training run writes.

Generic: any training that writes sample images (or other image eval artifacts) to a folder, local or over ssh.

  every poll_s
    list      the source folder (local walk or `ssh ... find`); a regex on each relative path gives
              step (+ prompt, variant): e.g. ai-toolkit `1793019187897__000002000_9.jpg` or `500/p005.jpg`
    detect    a step is a complete set when it has `expect` images (or no new file for `settle_s`)
    fetch     ssh sets are copied into out_dir/cache/<step>/ (one tar stream)
    sheets    per prompt a row: REF (fixed reference, e.g. base renders) | PREV <step> | CUR <step> per variant,
              `rows_per_sheet` rows per sheet (sheet.py: ffmpeg + built-in font, no Pillow)
    judge     one `codex exec -m gpt-6.1-terra --image ... --output-schema` call -> JSON verdict
              {step, structure_score, look_score, vs_previous{structure,look}, vs_reference{...},
               previous_regressions_persist, regressions[], notable[], recommend, confidence, reason}
    guard     Codex credits read before/after each call (codex app-server, no model call); per-run cap
              (`credit_cap`); over it -> no more calls + one critical alert
    policy    deterministic stop rules over the verdict history (decide()); the model's own `recommend` is
              advice only and logged
    act       `act: "stop"` runs the configured stop command ONCE on a stop decision (never destroys anything)
    report    every verdict -> alerts.jsonl (the watcher's file and/or our own), verdicts.jsonl, judge-log.md
              (human-readable, sheets next to it), status.json

Finite: exits at `deadline_local` / `max_hours`, after `last_step` is judged, or after acting on a stop.
"""
from __future__ import annotations

import copy
import glob
import io
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import tarfile
import tempfile
import time

from . import config as cfgmod
from . import meter as codex_meter
from . import timeutil as tu
from .redact import deep as redact_deep
from .sheet import build_sheet

CODEX = os.environ.get("CODEX_BIN") or shutil.which("codex") or "/opt/homebrew/bin/codex"
CODEX_HOME = os.path.expanduser(os.environ.get("CODEX_HOME") or "~/.codex")
# Lean Codex setup: our instructions replace the coding-agent base instructions, every tool/feature off,
# read-only sandbox.
CODEX_DISABLE = ("apps", "plugins", "image_generation", "multi_agent", "browser_use", "computer_use", "shell_tool",
                 "unified_exec", "skill_search", "goals", "in_app_browser", "remote_plugin", "tool_suggest",
                 "sleep_tool")

DEFAULTS: dict = {
    "name": "eval",
    "timezone": None,
    "out_dir": "./eval-out",
    "alerts_jsonl": [],                  # extra alerts files (e.g. the live watcher's), besides out_dir/alerts.jsonl
    "lifecycle": {"max_hours": 14, "deadline_local": None, "last_step": None, "exit_after_act": True},
    "poll_s": 120,
    "source": {
        "ssh": None,                     # None = local; else argv prefix, e.g. ["ssh", "-o", "BatchMode=yes", "user@host"]
        "dir": None,
        "pattern": r"^\d+__(?P<step>\d+)_(?P<idx>\d+)\.(?:jpg|jpeg|png|webp)$",   # on the path relative to dir
        "group_size": 1,                 # with an `idx` group: prompt = idx // group_size, variant = idx % group_size
        "expect": None,                  # images per complete set; None = settle only
        "settle_s": 600,                 # expect None: judge a set when no new file arrived for this long
        "partial_after_s": 2700,         # expect set: judge a set below `expect` only when it is this old AND a
                                         # later step exists (then it is recommend-only, never acted on)
        "min_age_s": 20,                 # newest file at least this old (writer finished)
        "min_step": 1,                   # ignore sets below this step
        "steps": None,                   # replay / tests: only these steps
        "min_epoch": None,               # ignore files written before this (unix s or local time string)
        "launch_gap_s": 1800,            # a step written again this much later than before = a new launch / resume
        "check_integrity": True,         # JPEG must end with FFD9, PNG with IEND (rsync of a half-written file)
        "timeout_s": 120,
    },
    "reference": {"dir": None, "pattern": r"^p(?P<prompt>\d+)\.(?:jpg|jpeg|png|webp)$", "label": "REF"},
    "prompts": {"file": None, "list_key": "prompts", "key_field": "index", "label_field": "category",
                "text_field": "prompt", "look_for_field": "look_for", "text_chars": 260},
    "sheet": {"variants": [0, 1], "rows_per_sheet": 4, "tile": 288, "prompts": None, "quality": 3},
    "judge": {
        "model": "gpt-6.1-terra",
        "effort": "medium",
        "timeout_s": 900,
        "retries": 1,
        "credit_cap": 200,               # Codex credits per judge run (balance drop; an upper bound if other Codex
                                         # users share the account at the same time)
        "codex": None,
        "subject": "a text-to-image diffusion model being fine-tuned",
        "structure": "object structure and correctness: phones and camera hardware (lens count, camera bars, "
                     "screens), cars (wheels, doors, wings, lights), hands and limbs (finger count, merged or extra "
                     "limbs), text and signs (spelling, legibility), counts of people/objects asked for in the prompt, "
                     "merged or fused objects",
        "look": "casual smartphone photo realism: candid framing, real phone exposure and noise, "
                "harder natural light, real night exposure, clutter and grime, no glossy 'AI' sheen, no saturated "
                "golden-hour cast, no plastic skin",
        "extra_instructions": "",
    },
    "policy": {
        "regression_confirm": "persist",  # persist: 2nd set worse OR the judge says the regression persists (not better)
                                          # worse_twice: 2nd set must also be 'worse' vs its previous set
        "regress_consecutive": 2,
        "plateau_consecutive": 2,
        "plateau_min_step": 1500,        # plateau counts only sets with step > this
        "plateau_act": True,             # False: a plateau stop is recommend-only (critical alert)
        "min_confidence": 0.5,
        "stop_min_confidence": 0.7,      # every verdict behind a stop decision needs at least this
        "same_prompt": True,             # regression stop: both verdicts must name the same prompt number (#N)
        # before acting: one independent call (no earlier verdicts shown), all variants, zoomed rows of the named
        # prompts; act only when it agrees
        "confirm": {"enabled": True, "effort": "high", "variants": None, "tile": 288, "zoom_tile": 576,
                    "zoom_rows_per_sheet": 2, "min_confidence": 0.7},
    },
    "act": "none",                       # none | stop
    "stop_command": {"argv": None, "cwd": None, "env": {}, "timeout_s": 180,
                     "attempts": 3, "retry_s": 180,   # rc != 0 -> retry; all failed -> critical alert, retry every poll
                     "precheck": None,                # argv run at start (e.g. ssh ... true): warn when it fails
                     "followup_check": None},         # argv: rc 0 = something will clean up after the stop
    "event_log": {"argv": None},         # optional argv prefix; every milestone is appended as one extra argument
}

SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["step", "structure_score", "look_score", "vs_previous", "vs_reference", "previous_regressions_persist",
                 "regressions", "notable", "recommend", "confidence", "reason"],
    "properties": {
        "step": {"type": "integer"},
        "structure_score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "look_score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "vs_previous": {"type": "object", "additionalProperties": False, "required": ["structure", "look"],
                        "properties": {"structure": {"type": "string", "enum": ["better", "same", "worse", "n/a"]},
                                       "look": {"type": "string", "enum": ["better", "same", "worse", "n/a"]}}},
        "vs_reference": {"type": "object", "additionalProperties": False, "required": ["structure", "look"],
                         "properties": {"structure": {"type": "string", "enum": ["better", "same", "worse", "n/a"]},
                                        "look": {"type": "string", "enum": ["better", "same", "worse", "n/a"]}}},
        "previous_regressions_persist": {"type": "string", "enum": ["yes", "partly", "no", "n/a"]},
        "regressions": {"type": "array", "items": {"type": "string"}},
        "notable": {"type": "array", "items": {"type": "string"}},
        "recommend": {"type": "string", "enum": ["continue", "stop"]},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
}
CMP = ("better", "same", "worse", "n/a")
CONFIRM_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["structure", "look", "regressions", "confidence", "reason"],
    "properties": {
        "structure": {"type": "string", "enum": ["better", "same", "worse"]},
        "look": {"type": "string", "enum": ["better", "same", "worse"]},
        "regressions": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
}


def load_config(path: str | None, overrides: dict | None = None) -> dict:
    user = cfgmod._read_chain(path) if path else {}
    cfg = cfgmod.deep_merge(DEFAULTS, user)
    cfg = cfgmod.deep_merge(cfg, overrides or {})
    cfg["_base"] = os.path.dirname(os.path.abspath(path)) if path else os.getcwd()
    cfg["out_dir"] = cfgmod.resolve(cfg, cfg["out_dir"])
    for sect, key in (("source", "dir"), ("reference", "dir"), ("prompts", "file"), ("stop_command", "cwd")):
        if cfg[sect].get(key) and not (sect == "source" and cfg["source"].get("ssh")):
            cfg[sect][key] = cfgmod.resolve(cfg, cfg[sect][key])
    cfg["alerts_jsonl"] = [cfgmod.resolve(cfg, p) for p in (cfg.get("alerts_jsonl") or [])]
    return cfg


# ---------------------------------------------------------------- listing / set detection

def parse_name(rel: str, pattern: str, group_size: int = 1):
    """Relative path -> (step, prompt, variant) or None. Groups: step (required); prompt + variant, or idx."""
    m = re.search(pattern, rel)
    if not m:
        return None
    g = m.groupdict()
    if g.get("step") is None:
        return None
    step = int(g["step"])
    if g.get("prompt") is not None:
        prompt = _num(g["prompt"])
        variant = _num(g["variant"]) if g.get("variant") is not None else 0
    elif g.get("idx") is not None:
        idx, gs = int(g["idx"]), max(1, int(group_size or 1))
        prompt, variant = idx // gs, idx % gs
    else:
        return None
    return step, prompt, variant


def _num(s):
    return int(s) if re.fullmatch(r"\d+", str(s)) else s


def list_local(root: str) -> list[tuple[str, int, float]]:
    out = []
    for dp, _dn, fns in os.walk(root):
        for fn in fns:
            p = os.path.join(dp, fn)
            try:
                st = os.stat(p)
            except OSError:
                continue
            out.append((os.path.relpath(p, root), st.st_size, st.st_mtime))
    return out


def list_ssh(ssh: list[str], root: str, timeout: float = 120) -> list[tuple[str, int, float]]:
    cmd = f"find {shlex.quote(root)} -type f -printf '%P\\t%s\\t%T@\\n' 2>/dev/null || true"
    r = subprocess.run([*ssh, cmd], capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    if r.returncode not in (0,):
        raise RuntimeError(f"ssh list failed rc={r.returncode}: {(r.stderr or '')[-200:]}")
    out = []
    for line in r.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) == 3:
            try:
                out.append((parts[0], int(parts[1]), float(parts[2])))
            except ValueError:
                pass
    return out


def file_ts(rel: str, mtime: float, pattern: str) -> float:
    """Write time of a file: the pattern's `ts` group (epoch ms or s, e.g. ai-toolkit's name prefix), else mtime."""
    m = re.search(pattern, rel)
    if m and m.groupdict().get("ts"):
        t = float(m.group("ts"))
        return t / 1000.0 if t > 1e11 else t
    return mtime


def _min_epoch(src: dict) -> float:
    v = src.get("min_epoch")
    if v in (None, ""):
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return tu.parse_local(str(v)) or 0.0


def launch_filter(parsed: list, gap: float) -> tuple[list, float | None]:
    """parsed [(ts, step, prompt, variant, rel, size, mtime)] -> (files of the current launch, launch id).

    Files sorted by write time form segments; a new segment starts when a step that is <= the highest step of the
    current segment is written more than `gap` s after that step's newest file (a restart or an automatic resume).
    A segment whose first step is <= the first step of the chain so far is a restart from scratch: everything
    before it is dropped and the launch id becomes its start time. Otherwise it is a resume: steps below its first
    step are kept from before, and its files replace the older ones."""
    if not parsed:
        return [], None
    parsed = sorted(parsed)
    if not gap or gap <= 0:                     # launch detection off (e.g. renders not written in step order)
        return parsed, None
    kept: list = []
    seg: list = []
    seg_newest: dict[int, float] = {}
    launch = parsed[0][0]
    chain_first = parsed[0][1]
    for f in parsed:
        ts, step = f[0], f[1]
        top = max(seg_newest) if seg_newest else None
        if top is not None and step <= top and ts - max(t for st, t in seg_newest.items() if st >= step) > gap:
            kept += seg
            if step <= chain_first:            # restart from scratch
                kept, launch, chain_first = [], ts, step
            else:                              # resume: older files at steps >= step are superseded
                kept = [k for k in kept if k[1] < step]
            seg, seg_newest = [], {}
        seg.append(f)
        seg_newest[step] = max(seg_newest.get(step, 0.0), ts)
    return kept + seg, launch


def group_sets(files, src: dict, with_launch: bool = False):
    """files [(rel, size, mtime)] -> {step: {"files": {(prompt, variant): rel}, "meta": {(prompt, variant): (size,
    mtime)}, "newest": write time, "n": count}} for the current launch (see launch_filter)."""
    parsed = []
    floor = _min_epoch(src)
    for rel, size, mt in files:
        if size <= 0:
            continue
        k = parse_name(rel, src["pattern"], src.get("group_size", 1))
        if not k:
            continue
        step, prompt, variant = k
        if step < int(src.get("min_step") or 0) or (src.get("steps") and step not in src["steps"]):
            continue
        ts = file_ts(rel, mt, src["pattern"])
        if ts < floor:
            continue
        parsed.append((ts, step, prompt, variant, rel, size, mt))
    gap = src.get("launch_gap_s", 1800)
    cur, launch = launch_filter(parsed, float(gap or 0))
    sets: dict[int, dict] = {}
    for ts, step, prompt, variant, rel, size, mt in cur:      # sorted by ts: the newest file per key wins
        s = sets.setdefault(step, {"files": {}, "meta": {}, "newest": 0.0})
        s["files"][(prompt, variant)] = rel
        s["meta"][(prompt, variant)] = (size, mt)
        s["newest"] = max(s["newest"], ts, mt)
    for s in sets.values():
        s["n"] = len(s["files"])
    return (sets, launch) if with_launch else sets


def ready_steps(sets: dict, src: dict, now: float, done: set) -> list[int]:
    """Steps to judge now. With `expect`: complete sets; a set below `expect` only when it is `partial_after_s` old
    and a later step exists. Without: when no new file arrived for `settle_s`."""
    out = []
    for step in sorted(sets):
        if step in done:
            continue
        s = sets[step]
        age = now - s["newest"]
        if age < float(src.get("min_age_s") or 0):
            continue
        exp = src.get("expect")
        if exp:
            later = any(x > step for x in sets)
            if len(s["files"]) >= int(exp) or (later and age >= float(src.get("partial_after_s") or 2700)):
                out.append(step)
        elif age >= float(src.get("settle_s") or 600):
            out.append(step)
    return out


def image_complete(path: str) -> bool:
    """False for an obviously truncated image (copied mid-write): JPEG without the FFD9 end marker, PNG without
    IEND. Other types: non-empty."""
    try:
        size = os.path.getsize(path)
        if size < 16:
            return False
        with open(path, "rb") as f:
            head = f.read(8)
            f.seek(max(0, size - 64))
            tail = f.read().rstrip(b"\x00")
    except OSError:
        return False
    if head[:2] == b"\xff\xd8":
        return tail.endswith(b"\xff\xd9")
    if head == b"\x89PNG\r\n\x1a\n":
        return b"IEND" in tail[-16:]
    return True


def fetch_ssh(ssh: list[str], root: str, rels: list[str], dest: str, timeout: float = 300) -> int:
    """Copy rels (relative to root on the remote) into dest with one tar stream. Returns files written."""
    os.makedirs(dest, exist_ok=True)
    want = set(rels)
    cmd = f"tar -C {shlex.quote(root)} -cf - " + " ".join(shlex.quote(r) for r in rels)
    r = subprocess.run([*ssh, cmd], capture_output=True, timeout=timeout, stdin=subprocess.DEVNULL)
    if r.returncode or not r.stdout:
        raise RuntimeError(f"ssh tar failed rc={r.returncode}: {r.stderr.decode(errors='replace')[-200:]}")
    n = 0
    with tarfile.open(fileobj=io.BytesIO(r.stdout)) as tf:
        for m in tf.getmembers():
            if not m.isfile() or m.name not in want:
                continue
            out = os.path.join(dest, m.name.replace("/", "__"))
            data = tf.extractfile(m).read()
            with open(out + ".part", "wb") as f:
                f.write(data)
            os.replace(out + ".part", out)
            n += 1
    return n


def load_prompts(pc: dict) -> dict:
    """{prompt_key: {"label": str, "text": str, "look_for": str}} from an optional JSON file."""
    if not pc.get("file") or not os.path.exists(pc["file"]):
        return {}
    with open(pc["file"]) as f:
        d = json.load(f)
    items = d.get(pc.get("list_key") or "prompts", []) if isinstance(d, dict) else d
    out = {}
    for i, it in enumerate(items):
        if isinstance(it, str):
            out[i] = {"label": it[:80], "text": it, "look_for": ""}
            continue
        key = _num(it.get(pc.get("key_field") or "index", i))
        text = it.get(pc.get("text_field") or "prompt") or it.get("caption") or ""
        if isinstance(text, str) and text.lstrip().startswith("{"):
            try:
                text = json.loads(text).get("short_description") or text
            except ValueError:
                pass
        out[key] = {"label": str(it.get(pc.get("label_field") or "category") or f"prompt {key}"),
                    "text": str(text)[:int(pc.get("text_chars") or 260)],
                    "look_for": str(it.get(pc.get("look_for_field") or "look_for") or "")}
    return out


# ---------------------------------------------------------------- policy

def _ok(v, minc):
    try:
        return float(v.get("confidence") or 0) >= minc
    except (TypeError, ValueError):
        return False


def prompt_nums(items) -> set[int]:
    """Prompt numbers named as '#N' in a list of strings."""
    return {int(x) for it in (items or []) for x in re.findall(r"#(\d+)", str(it))}


_RANK = {"worse": 0, "same": 1, "better": 2}


def decide(history: list[dict], pol: dict) -> dict:
    """Deterministic stop policy over the judged verdicts (oldest first). Returns
    {"decision": "continue"|"stop", "rule", "why", "steps", "before_step", "prompts"}."""
    minc = max(float(pol.get("min_confidence", 0.5)), float(pol.get("stop_min_confidence", 0.0)))
    if not history:
        return {"decision": "continue", "rule": None, "why": "no verdicts yet"}
    # (a) structure regression vs the previous set, confirmed on the next set
    n = max(2, int(pol.get("regress_consecutive", 2)))
    tail = history[-n:]
    if len(tail) == n and all(_ok(v, minc) for v in tail):
        first, rest = tail[0], tail[1:]
        if first["vs_previous"]["structure"] == "worse":
            mode = pol.get("regression_confirm", "persist")
            confirmed = all(v["vs_previous"]["structure"] == "worse" or
                            (mode == "persist" and v.get("previous_regressions_persist") == "yes" and
                             v["vs_previous"]["structure"] != "better") for v in rest)
            common = prompt_nums(first.get("regressions"))
            for v in rest:
                common &= prompt_nums(v.get("regressions"))
            if confirmed and pol.get("same_prompt", True) and not common:
                return {"decision": "continue", "rule": None,
                        "why": f"structure worse at {first['step']} and again at {tail[-1]['step']}, but the "
                               f"regressions do not name the same prompt in both verdicts"}
            if confirmed:
                steps = ", ".join(str(v["step"]) for v in tail)
                return {"decision": "stop", "rule": "structure_regression",
                        "steps": [v["step"] for v in tail], "before_step": first.get("prev_step"),
                        "prompts": sorted(common),
                        "why": f"structure got worse at step {first['step']} and was confirmed on the next set "
                               f"(steps {steps}): " + "; ".join((first.get("regressions") or [])[:3])}
    # (b) plateau: no improvement in structure or look for k consecutive sets with step > plateau_min_step; the
    # scores and the vs-reference calls did not rise either (vs the set before the window)
    k = max(1, int(pol.get("plateau_consecutive", 2)))
    tail = history[-k:]
    base = history[-k - 1] if len(history) > k else None
    if base is not None and len(tail) == k and all(v["step"] > int(pol.get("plateau_min_step", 0)) for v in tail) \
            and all(_ok(v, minc) for v in tail) \
            and all(v["vs_previous"]["structure"] in ("same", "worse") and v["vs_previous"]["look"] in ("same", "worse")
                    for v in tail) \
            and all((v.get("structure_score") or 0) <= (base.get("structure_score") or 0) and
                    (v.get("look_score") or 0) <= (base.get("look_score") or 0) for v in tail) \
            and all(_RANK.get(v["vs_reference"][a], -1) <= _RANK.get(base["vs_reference"][a], 9)
                    for v in tail for a in ("structure", "look")):
        return {"decision": "stop", "rule": "plateau", "steps": [v["step"] for v in tail],
                "before_step": base["step"], "prompts": [],
                "why": f"no improvement in structure or look for {k} consecutive sets (steps "
                       f"{', '.join(str(v['step']) for v in tail)}) after update {pol.get('plateau_min_step')}; "
                       f"scores and vs-reference not above step {base['step']}"}
    last = history[-1]
    trend = f"structure {last['vs_previous']['structure']}, look {last['vs_previous']['look']} vs previous"
    return {"decision": "continue", "rule": None, "why": trend}


def normalize(v: dict, step: int) -> tuple[dict, list[str]]:
    """Clamp / fill a model verdict; returns (verdict, problems)."""
    probs = []
    out = copy.deepcopy(v) if isinstance(v, dict) else {}
    if out.get("step") != step:
        if out.get("step") is not None:
            probs.append(f"model said step {out.get('step')}, set is {step}")
        out["step"] = step
    for k in ("structure_score", "look_score"):
        try:
            out[k] = min(5, max(1, int(out.get(k))))
        except (TypeError, ValueError):
            probs.append(f"bad {k}")
            out[k] = None
    for k in ("vs_previous", "vs_reference"):
        d = out.get(k) if isinstance(out.get(k), dict) else {}
        out[k] = {a: (d.get(a) if d.get(a) in CMP else "n/a") for a in ("structure", "look")}
    if out.get("previous_regressions_persist") not in ("yes", "partly", "no", "n/a"):
        out["previous_regressions_persist"] = "n/a"
    for k in ("regressions", "notable"):
        out[k] = [str(x)[:300] for x in (out.get(k) or [])][:12]
    if out.get("recommend") not in ("continue", "stop"):
        out["recommend"] = "continue"
    try:
        c = float(out.get("confidence"))
        out["confidence"] = round(c / 100 if c > 1 else max(0.0, c), 3)
    except (TypeError, ValueError):
        out["confidence"] = 0.0
        probs.append("bad confidence")
    out["reason"] = str(out.get("reason") or "")[:1200]
    return out, probs


# ---------------------------------------------------------------- Codex call

def rollout_info(thread_id: str):
    """(balance, primary_pct, token_usage) from the Codex session file of this call (None if missing)."""
    fs = glob.glob(os.path.join(CODEX_HOME, "sessions", "*", "*", "*", f"rollout-*{thread_id}.jsonl"))
    rl, usage = None, None
    for f in fs:
        for line in open(f, errors="replace"):
            if '"rate_limits"' in line or '"token_count"' in line:
                try:
                    p = json.loads(line).get("payload", {})
                except ValueError:
                    continue
                rl = p.get("rate_limits") or rl
                usage = (p.get("info") or {}).get("total_token_usage") or usage
    if not rl:
        return None, None, usage
    cr = rl.get("credits") or {}
    try:
        bal = float(cr.get("balance"))
    except (TypeError, ValueError):
        bal = None
    return bal, (rl.get("primary") or {}).get("used_percent"), usage


def codex_judge(images: list[str], prompt: str, instructions: str, jc: dict, scratch: str) -> dict:
    """One `codex exec` call. Returns {"ok", "verdict"|"error", "secs", "thread_id", "usage", "balance_after",
    "pct_after"}."""
    d = tempfile.mkdtemp(prefix="evaljudge-", dir=scratch)
    try:
        schema, out, instr = (os.path.join(d, x) for x in ("schema.json", "out.json", "instructions.txt"))
        with open(schema, "w") as f:
            json.dump(jc.get("_schema") or SCHEMA, f)
        with open(instr, "w") as f:
            f.write(instructions)
        imgs = [x for p in images for x in ("--image", p)]
        lean = ["-c", f'model_instructions_file="{instr}"', "-c", "skills.include_instructions=false",
                *[x for f in CODEX_DISABLE for x in ("--disable", f)]]
        cmd = [jc.get("codex") or CODEX, "exec", "-m", jc["model"], "-c", f'model_reasoning_effort="{jc["effort"]}"',
               "-c", 'approval_policy="never"', *lean, "-s", "read-only", "--skip-git-repo-check", "--json", "-C", d,
               *imgs, "--output-schema", schema, "-o", out, "--", prompt]
        t = time.time()
        r = subprocess.run(cmd, cwd=d, capture_output=True, text=True, timeout=float(jc.get("timeout_s", 900)),
                           stdin=subprocess.DEVNULL)
        secs = round(time.time() - t, 1)
        tid = None
        for line in (r.stdout or "").splitlines():
            try:
                tid = tid or json.loads(line).get("thread_id")
            except ValueError:
                pass
        bal, pct, usage = rollout_info(tid) if tid else (None, None, None)
        res = {"secs": secs, "thread_id": tid, "usage": usage, "balance_after": bal, "pct_after": pct}
        if r.returncode or not os.path.exists(out):
            err = ((r.stderr or "")[-400:] + " | " + (r.stdout or "")[-300:]).strip()
            return dict(res, ok=False, error=f"rc={r.returncode}: {err}"[-700:])
        try:
            with open(out) as f:
                return dict(res, ok=True, verdict=json.load(f))
        except ValueError as e:
            return dict(res, ok=False, error=f"bad JSON from model: {e}")
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout after {jc.get('timeout_s')} s", "secs": float(jc.get("timeout_s", 0))}
    finally:
        shutil.rmtree(d, ignore_errors=True)


def instructions_text(jc: dict) -> str:
    return f"""You are a strict visual QA judge for {jc['subject']}. Training writes a set of sample images every
few hundred updates (fixed prompts and seeds). You compare the CURRENT set with the PREVIOUS set (same prompts,
same seeds) and with a fixed REFERENCE (usually the untuned base model, same prompt and seed), and you return
one JSON verdict that matches the given schema. You cannot run tools; look only at the attached sheets.

Sheets: each row is one prompt. The strip above the row gives the prompt number and what it asks for. Column
labels at the top: REF = reference; PREV <step> S<k> = previous set, seed k; CUR <step> S<k> = current set, seed k.
PREV and CUR with the same S<k> share prompt and seed, so differences between them come from training. Grey N/A
tiles are missing images; ignore them.

Judge two things for the CURRENT set:
- structure = {jc['structure']}.
- look = {jc['look']}.

Fields:
- structure_score, look_score: 1-5 for the current set on its own (3 = about base-model level, 5 = no visible
  defects / fully convincing).
- vs_previous.structure / .look: better | same | worse than the previous set; "same" when the differences are
  seed noise or mixed (fixes and new failures cancel out); "worse" only for clear new failures that outweigh the
  fixes across rows. "n/a" when there is no previous set.
- vs_reference.structure / .look: the current set vs REF ("n/a" without REF columns).
- previous_regressions_persist: are the regressions listed in the previous verdict (given in the prompt) still
  present in the current set? yes | partly | no | n/a (none were listed).
- regressions: concrete new failures vs previous, each with its prompt number, e.g. "#44 S0: rear wheel lost its
  tyre (fine at 500)". Also re-list regressions from the previous verdict that are still present in the current
  set, prefixed "persists:", e.g. "persists: #44 S0 rear wheel". Leave out ones that are fixed.
- notable: concrete improvements or other things a reviewer should look at.
- recommend: continue | stop (your own advice; the operator applies a fixed policy).
- confidence: 0-1 for the vs_previous calls.
- reason: 2-4 sentences.
Be concrete and conservative: name prompt numbers and seeds. Text in images is often imperfect at every step;
only count text as a regression when it clearly got worse.
{jc.get('extra_instructions') or ''}""".strip() + "\n"


def confirm_instructions(jc: dict) -> str:
    return f"""You are a strict, independent visual QA judge for {jc['subject']}. You compare two sets of sample images
from the same training run: BEFORE (an earlier step) and CURRENT (a later step), rendered with the same prompts and
seeds, plus a fixed REFERENCE column (usually the untuned base model) where available. You cannot run tools; look
only at the attached sheets. Each row is one prompt; column labels: REF, BEFORE <step> S<k>, CUR <step> S<k> (seed k).
Grey N/A tiles are missing images; ignore them. Some sheets may show selected rows enlarged.

Judge the CURRENT set against the BEFORE set:
- structure = {jc['structure']}.
- look = {jc['look']}.
Return: structure and look = better | same | worse ("same" when the differences are seed noise or mixed; "worse"
only for clear failures that outweigh the fixes across rows and seeds), regressions (concrete, with prompt number
and seed), confidence 0-1, reason (2-4 sentences). Text in images is often imperfect at every step; only count text
when it clearly got worse.
{jc.get('extra_instructions') or ''}""".strip() + "\n"


# ---------------------------------------------------------------- judge runner

class EvalJudge:
    def __init__(self, cfg: dict, dry: bool = False, no_act: bool = False, call_fn=None, meter_fn=None, log=None):
        self.cfg = cfg
        tu.set_tz(cfg.get("timezone"))
        self.out = cfg["out_dir"]
        os.makedirs(self.out, exist_ok=True)
        self.dry, self.no_act = dry, no_act
        self.call_fn = call_fn or codex_judge
        self.meter_fn = meter_fn or codex_meter.read_summary
        self._log = log
        self.state_path = os.path.join(self.out, "state.json")
        self.st = {"judged": {}, "failed": {}, "history": [], "credits": {"start": None, "used": 0.0, "calls": 0,
                   "per_call": [], "tokens": 0, "capped": False}, "acted": None, "decisions": [], "started": None,
                   "cache": {}}
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path) as f:
                    self.st.update(json.load(f))
            except ValueError:
                pass
        self.prompts = load_prompts(cfg["prompts"])
        self.ref = self._load_ref()

    # -- io helpers
    def log(self, msg: str):
        line = f"{tu.stamp(time.time())} {msg}"
        if self._log:
            self._log(line)
        with open(os.path.join(self.out, "eval-judge.log"), "a") as f:
            f.write(line + "\n")

    def save(self):
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.st, f, indent=1, default=str)
        os.replace(tmp, self.state_path)

    def status(self, **kw):
        s = {"t": time.time(), "time": tu.stamp(time.time()), "name": self.cfg["name"], "pid": os.getpid(),
             "judged": sorted(int(k) for k in self.st["judged"]), "credits_used": round(self.st["credits"]["used"], 3),
             "credit_cap": self.cfg["judge"]["credit_cap"], "acted": self.st.get("acted")}
        s.update(kw)
        tmp = os.path.join(self.out, "status.json.tmp")
        with open(tmp, "w") as f:
            json.dump(s, f, indent=1, default=str)
        os.replace(tmp, os.path.join(self.out, "status.json"))

    def alert(self, severity: str, what: str, rule: str = "eval_judge", evidence=None, action=None, **extra):
        now = time.time()
        rec = {"t": now, "time": tu.fmt12(now, date=True), "status": "new", "severity": severity, "rule": rule,
               "key": f"{rule}:{extra.get('step', '')}", "what": what, "evidence": list(evidence or [])[:8],
               "action": action, "since": tu.fmt12(now), "count": 1, "source": "eval_judge"}
        rec.update(extra)
        rec = redact_deep(rec)
        line = json.dumps(rec, default=str) + "\n"
        for p in [os.path.join(self.out, "alerts.jsonl"), *self.cfg.get("alerts_jsonl", [])]:
            try:
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "a") as f:
                    f.write(line)
            except OSError as e:
                self.log(f"alert write failed {p}: {e}")
        return rec

    def event_log(self, msg: str):
        argv = (self.cfg.get("event_log") or {}).get("argv")
        if not argv or self.dry:
            return
        try:
            subprocess.run([*argv, msg[:400]], capture_output=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            pass

    # -- data
    def _load_ref(self) -> dict:
        rc = self.cfg["reference"]
        if not rc.get("dir") or not os.path.isdir(rc["dir"]):
            return {}
        out = {}
        for rel, size, _mt in list_local(rc["dir"]):
            m = re.search(rc["pattern"], rel)
            if not m or size <= 0:
                continue
            g = m.groupdict()
            key = (_num(g["prompt"]), _num(g["variant"]) if g.get("variant") is not None else 0)
            out[key] = os.path.join(rc["dir"], rel)
        return out

    def local_paths(self, step: int, s: dict) -> dict:
        """{(prompt, variant): local path}; fetches ssh sets into out_dir/cache/<step>/. A cached copy is fetched
        again when the source file's (size, mtime) changed (e.g. rsync replaced a half-copied file)."""
        src = self.cfg["source"]
        if not src.get("ssh"):
            return {k: os.path.join(src["dir"], rel) for k, rel in s["files"].items()}
        dest = os.path.join(self.out, "cache", str(step))
        cache = self.st.setdefault("cache", {})
        local = {k: os.path.join(dest, rel.replace("/", "__")) for k, rel in s["files"].items()}
        sig = {k: list(s.get("meta", {}).get(k, (0, 0))) for k in local}
        need = [k for k, p in local.items() if not os.path.exists(p) or cache.get(p) != sig[k]]
        if need:
            fetch_ssh(src["ssh"], src["dir"], [s["files"][k] for k in need], dest,
                      timeout=max(300, src.get("timeout_s", 120)))
            for k in need:
                if os.path.exists(local[k]):
                    cache[local[k]] = sig[k]
        return {k: p for k, p in local.items() if os.path.exists(p)}

    def usable(self, step: int, s: dict, now: float) -> tuple[dict | None, list]:
        """Local paths of the set's complete images, or None = wait a poll (some images are truncated and the set is
        still young enough that rsync will replace them). Returns (paths, bad keys)."""
        paths = self.local_paths(step, s)
        src = self.cfg["source"]
        if not src.get("check_integrity", True):
            return paths, []
        bad = [k for k, p in paths.items() if not image_complete(p)]
        if bad:
            for k in bad:
                self.st.get("cache", {}).pop(paths[k], None)          # fetch again next time
            if now - s["newest"] < float(src.get("partial_after_s") or 2700):
                self.log(f"step {step}: {len(bad)} truncated images (e.g. {s['files'][bad[0]]}); waiting a poll")
                return None, bad
            self.log(f"step {step}: {len(bad)} images still truncated after {src.get('partial_after_s')} s; "
                     f"judging without them (recommend-only)")
            paths = {k: p for k, p in paths.items() if k not in bad}
        return paths, bad

    # -- sheets + prompt
    def make_sheets(self, step: int, cur: dict, prev_step: int | None, prev: dict | None, variants=None,
                    prompts=None, tile=None, rows_per_sheet=None, subdir=None, prev_label="PREV",
                    cur_label="CUR") -> list[str]:
        sc = self.cfg["sheet"]
        variants = variants or sc.get("variants") or sorted({v for _p, v in cur})
        prompts = prompts or sc.get("prompts") or sorted({p for p, _v in cur}, key=lambda x: (str(type(x)), x))
        has_ref = any((p, 0) in self.ref for p in prompts)
        cols = ([self.cfg["reference"].get("label", "REF")] if has_ref else [])
        for v in variants:
            if prev is not None:
                cols.append(f"{prev_label} {prev_step} S{v}")
            cols.append(f"{cur_label} {step} S{v}")
        rows = []
        for p in prompts:
            meta = self.prompts.get(p, {})
            paths = ([self.ref.get((p, 0))] if has_ref else [])
            for v in variants:
                if prev is not None:
                    paths.append(prev.get((p, v)))
                paths.append(cur.get((p, v)))
            rows.append((f"#{p} {meta.get('label', '')}", paths))
        n = max(1, int(rows_per_sheet or sc.get("rows_per_sheet") or 4))
        chunks = [rows[i:i + n] for i in range(0, len(rows), n)]
        d = os.path.join(self.out, "sheets", f"{step:06d}" if isinstance(step, int) else str(step))
        if subdir:
            d = os.path.join(d, subdir)
        os.makedirs(d, exist_ok=True)
        out = []
        for i, ch in enumerate(chunks, 1):
            title = (f"step {step}" + (f" vs {prev_step}" if prev is not None else "") +
                     f" | sheet {i}/{len(chunks)} | {self.cfg['name']}")
            p = os.path.join(d, f"sheet-{i}.jpg")
            build_sheet(ch, cols, p, title=title, tile=int(tile or sc.get("tile") or 256),
                        quality=int(sc.get("quality") or 3))
            out.append(p)
        return out

    def _prompt_lines(self, prompts) -> list[str]:
        lines = []
        for p in prompts:
            m = self.prompts.get(p)
            if m:
                extra = f" [check: {m['look_for']}]" if m.get("look_for") else ""
                lines.append(f"#{p} {m['label']}: {m['text']}{extra}")
            else:
                lines.append(f"#{p}")
        return lines

    def prompt_text(self, step: int, prev_step: int | None, n_sheets: int, cur: dict) -> str:
        sc = self.cfg["sheet"]
        prompts = sc.get("prompts") or sorted({p for p, _v in cur}, key=lambda x: (str(type(x)), x))
        lines = [f"Run: {self.cfg['name']}. CURRENT set = step {step}. "
                 + (f"PREVIOUS set = step {prev_step}." if prev_step is not None else
                    "There is no previous set: use n/a for vs_previous and previous_regressions_persist."),
                 f"{n_sheets} sheets attached, in order. Set `step` to {step}.", "", "Prompts (what each row asked for):"]
        lines += self._prompt_lines(prompts)
        last = self.st["history"][-1] if self.st["history"] else None
        if last and prev_step is not None and last["step"] == prev_step:
            lines += ["", f"Previous verdict (step {last['step']}): structure {last['structure_score']}, look "
                          f"{last['look_score']}, vs its previous: structure {last['vs_previous']['structure']}, look "
                          f"{last['vs_previous']['look']}.",
                      "Regressions it listed: " + ("; ".join(last.get("regressions") or []) or "none")]
        return "\n".join(lines)

    # -- credits
    def credits_ok(self) -> bool:
        c = self.st["credits"]
        cap = float(self.cfg["judge"]["credit_cap"])
        last = max(c["per_call"][-3:] or [0.0])
        if c["used"] >= cap or c["used"] + last > cap:
            if not c["capped"]:
                c["capped"] = True
                self.alert("critical", f"eval judge: Codex credit cap reached ({c['used']:.2f} of {cap:g} used); "
                                       f"no more judge calls this run", rule="eval_judge_credit_cap",
                           action="Judge the remaining sample sets by hand, or raise judge.credit_cap and restart.")
                self.event_log(f"eval judge credit cap reached: {c['used']:.2f}/{cap:g}")
            return False
        return True

    def _call(self, images: list[str], text: str, instr: str, jc: dict, what: str) -> dict | None:
        """One model call with retries + credit accounting; None when the credit cap blocks it."""
        if not self.credits_ok():
            return None
        res = None
        for attempt in range(1 + int(jc.get("retries", 1))):
            before = self.meter_fn()
            res = self.call_fn(images, text, instr, jc, self.out)
            after = self.meter_fn()
            self._account(before, after, res)
            if res.get("ok"):
                break
            self.log(f"{what}: judge call failed (attempt {attempt + 1}): {res.get('error')}")
            if not self.credits_ok():
                break
        return res

    # -- one set
    def judge_step(self, step: int, sets: dict, cur: dict) -> dict | None:
        s = sets[step]
        cands = set(self.st["judged"]) | ({k for k, x in self.st["failed"].items() if x == "dry run"} if self.dry else set())
        prev_step = max((int(k) for k in cands if int(k) < step), default=None)
        prev = None
        if prev_step is not None and prev_step in sets:
            prev = self.local_paths(prev_step, sets[prev_step])
        elif prev_step is not None:
            prev_step = None
        sheets = self.make_sheets(step, cur, prev_step, prev)
        text = self.prompt_text(step, prev_step, len(sheets), cur)
        if self.dry:
            self.log(f"dry: step {step} vs {prev_step}: {len(sheets)} sheets, {len(cur)} images, no model call")
            return None
        jc = self.cfg["judge"]
        res = self._call(sheets, text, instructions_text(jc), jc, f"step {step}")
        if res is None:
            return None
        if not res.get("ok"):
            self.st["failed"][str(step)] = res.get("error", "no call")
            self.alert("warn", f"eval judge: model call failed for step {step}: {res.get('error', '')[:200]}",
                       rule="eval_judge_error", step=step,
                       action=f"Open the sheets by hand: {os.path.dirname(sheets[0])}")
            self.save()
            return None
        v, probs = normalize(res["verdict"], step)
        exp = self.cfg["source"].get("expect")
        v.update({"prev_step": prev_step, "sheets": sheets, "secs": res.get("secs"), "problems": probs,
                  "n_images": len(cur), "partial": bool(exp and len(cur) < int(exp)),
                  "credits_call": self.st["credits"]["per_call"][-1] if self.st["credits"]["per_call"] else None,
                  "t": time.time(), "time": tu.fmt12(time.time(), date=True)})
        return v

    def _account(self, before, after, res):
        c = self.st["credits"]
        c["calls"] += 1
        b0 = (before or {}).get("credits")
        b1 = (after or {}).get("credits")
        if b1 is None:
            b1 = res.get("balance_after")
        if c["start"] is None and b0 is not None:
            c["start"] = b0
        cost = max(0.0, b0 - b1) if (b0 is not None and b1 is not None) else 0.0
        c["per_call"].append(round(cost, 4))
        c["used"] = round(c["used"] + cost, 4)
        c["last_balance"] = b1
        c["pct_before"], c["pct_after"] = (before or {}).get("primary_pct"), (after or {}).get("primary_pct",
                                                                                               res.get("pct_after"))
        u = res.get("usage") or {}
        c["tokens"] += int(u.get("total_tokens") or 0)
        c["last_usage"] = u

    # -- report + act
    def record(self, v: dict, sets: dict | None = None):
        self.st["judged"][str(v["step"])] = {k: v[k] for k in ("structure_score", "look_score", "vs_previous",
                                                                "recommend", "confidence")}
        self.st["judged"][str(v["step"])]["sig"] = v.get("sig")
        self.st["history"] = [h for h in self.st["history"] if h["step"] != v["step"]]
        self.st["history"].append({**{k: v.get(k) for k in SCHEMA["required"]}, "prev_step": v.get("prev_step"),
                                   "partial": v.get("partial", False)})
        self.st["history"].sort(key=lambda x: x["step"])
        dec = decide(self.st["history"], self.cfg["policy"])
        v["decision"] = dec
        with open(os.path.join(self.out, "verdicts.jsonl"), "a") as f:
            f.write(json.dumps(v, default=str) + "\n")
        self._human_log(v, dec)
        sev = "critical" if dec["decision"] == "stop" else (
            "warn" if "worse" in (v["vs_previous"]["structure"], v["vs_previous"]["look"]) or v["recommend"] == "stop"
            else "info")
        what = (f"eval judge step {v['step']}" + (f" vs {v['prev_step']}" if v.get("prev_step") is not None else "") +
                (f" (PARTIAL: {v.get('n_images')} images)" if v.get("partial") else "") +
                f": structure {v['structure_score']}/5 ({v['vs_previous']['structure']}), look {v['look_score']}/5 "
                f"({v['vs_previous']['look']}); model says {v['recommend']} ({v['confidence']:.2f}); policy: "
                f"{dec['decision'].upper()}" + (f" [{dec['rule']}]" if dec.get("rule") else ""))
        act = None
        if dec["decision"] == "stop":
            act = f"STOP decision ({dec['rule']}): {dec['why']}"[:400] + " " + self._act_plan(dec)
        self.alert(sev, what, rule="eval_verdict", step=v["step"], evidence=(v.get("regressions") or [])[:4] +
                   [f"reason: {v['reason'][:300]}", f"sheets: {os.path.dirname(v['sheets'][0])}"], action=act,
                   verdict={k: v.get(k) for k in SCHEMA["required"]}, decision=dec)
        self.event_log(f"eval judge step {v['step']}: S{v['structure_score']} L{v['look_score']} "
                       f"vs prev {v['vs_previous']['structure']}/{v['vs_previous']['look']} -> {dec['decision']}"
                       f"; credits used {self.st['credits']['used']:.2f}")
        self.st["decisions"].append({"step": v["step"], **dec})
        if dec["decision"] == "stop":
            self.act(v, dec, sets or {})
        self.save()

    def _acting(self) -> bool:
        return self.cfg.get("act") == "stop" and not self.no_act and not self.dry

    def _act_plan(self, dec: dict) -> str:
        if not self._acting():
            return "act=none: stop by hand if you agree."
        if dec.get("rule") == "plateau" and not self.cfg["policy"].get("plateau_act", True):
            return "Plateau stops are recommend-only (plateau_act false): stop by hand if you agree."
        if (self.cfg["policy"].get("confirm") or {}).get("enabled", True):
            return ("The judge now makes one independent confirmation call (all seeds, zoomed rows) and runs the "
                    "stop command only if it agrees.")
        return "The judge will run the configured stop command."

    def _recommend_only(self, v, dec, why: str):
        self.st.setdefault("recommended_stop", []).append({"step": v["step"], "rule": dec["rule"], "why": why})
        if why != "act=none":
            self.alert("critical", f"eval judge: STOP decision at step {v['step']} ({dec['rule']}) NOT acted on: {why}",
                       rule="eval_judge_act", step=v["step"], action="Review the sheets and stop by hand if you agree.")
            self._log_md(f"**{tu.fmt12(time.time(), date=True)}: stop decision at step {v['step']} not acted on**: {why}")

    def act(self, v: dict, dec: dict, sets: dict):
        if self.st.get("acted") or self.st.get("pending_stop"):
            return
        if not self._acting():
            # recommendation only: keep judging the next sets (the human / main agent decides)
            self._recommend_only(v, dec, "act=none")
            return
        hist = {h["step"]: h for h in self.st["history"]}
        if any(hist.get(st, {}).get("partial") for st in dec.get("steps") or [v["step"]]):
            self._recommend_only(v, dec, "a set behind it was incomplete (below `expect`)")
            return
        if dec.get("rule") == "plateau" and not self.cfg["policy"].get("plateau_act", True):
            self._recommend_only(v, dec, "plateau stops are recommend-only (plateau_act false)")
            return
        sc = self.cfg["stop_command"]
        if not sc.get("argv"):
            self.st["acted"] = {"step": v["step"], "ran": False, "why": "no stop_command configured"}
            self.save()
            self.alert("critical", "eval judge decided STOP but no stop_command is configured", rule="eval_judge_act",
                       step=v["step"], action="Stop the run by hand.")
            return
        if (self.cfg["policy"].get("confirm") or {}).get("enabled", True):
            ok, why = self.confirm(v, dec, sets)
            self.st.setdefault("confirmations", []).append({"step": v["step"], "rule": dec["rule"], "agree": ok,
                                                            "why": why})
            if not ok:
                self.alert("warn", f"eval judge: stop decision at step {v['step']} ({dec['rule']}) NOT confirmed by the "
                                   f"independent check; continuing: {why}"[:500], rule="eval_judge_confirm",
                           step=v["step"])
                self._log_md(f"**{tu.fmt12(time.time(), date=True)}: confirmation check disagreed, run continues**: "
                             f"{why}")
                self.event_log(f"eval judge stop at {v['step']} vetoed by confirmation: {why}"[:300])
                return
            self._log_md(f"**{tu.fmt12(time.time(), date=True)}: confirmation check agreed**: {why}")
        self.st["pending_stop"] = {"step": v["step"], "rule": dec["rule"], "why": dec["why"][:300], "attempts": 0,
                                   "t": time.time()}
        self.save()
        self.try_stop(int(sc.get("attempts") or 3))

    def try_stop(self, attempts: int) -> bool:
        """Run the stop command (idempotent: it only asks the trainer to save + exit) up to `attempts` times,
        `retry_s` apart. Success -> st['acted'] (the judge then exits). All failed -> critical alert once; the
        pending stop is retried every poll."""
        ps = self.st.get("pending_stop")
        if not ps or self.st.get("acted"):
            return bool(self.st.get("acted"))
        sc = self.cfg["stop_command"]
        env = dict(os.environ, **{k: str(x) for k, x in (sc.get("env") or {}).items()})
        rc, outp = -1, ""
        for i in range(max(1, attempts)):
            if i:
                self._sleep(float(sc.get("retry_s", 180)))
            try:
                r = subprocess.run(sc["argv"], cwd=sc.get("cwd") or None, env=env, capture_output=True, text=True,
                                   timeout=float(sc.get("timeout_s", 180)), stdin=subprocess.DEVNULL)
                rc, outp = r.returncode, ((r.stdout or "") + (r.stderr or ""))[-600:]
            except (OSError, subprocess.TimeoutExpired) as e:
                rc, outp = -1, f"{type(e).__name__}: {e}"
            ps["attempts"] += 1
            ps["last_rc"], ps["last_output"], ps["last_t"] = rc, outp[-300:], time.time()
            self.save()
            self.log(f"stop command attempt {ps['attempts']}: rc={rc}")
            if rc == 0:
                break
        step = ps["step"]
        if rc == 0:
            self.st["acted"] = {"step": step, "ran": True, "rc": 0, "t": time.time(), "argv": sc["argv"],
                                "attempts": ps["attempts"], "output": outp, "rule": ps["rule"]}
            self.st["pending_stop"] = None
            follow = self._followup()
            self.alert("critical", f"eval judge RAN the stop command at step {step} ({ps['rule']}): rc=0 after "
                                   f"{ps['attempts']} attempt(s)", rule="eval_judge_act", step=step,
                       evidence=[outp[-300:]], action=follow)
            self.event_log(f"eval judge ran stop command at step {step} ({ps['rule']}), rc=0; {follow}"[:400])
            self._log_md(f"**{tu.fmt12(time.time(), date=True)}: ran the stop command** `{' '.join(sc['argv'])}` -> "
                         f"rc 0 (attempt {ps['attempts']})\n\n{follow}\n\n```\n{outp[-400:]}\n```")
            self.save()
            return True
        if not ps.get("alerted"):
            ps["alerted"] = True
            self.alert("critical", f"eval judge: the stop command FAILED {ps['attempts']} times at step {step} "
                                   f"({ps['rule']}), last rc={rc}; retrying every poll", rule="eval_judge_act",
                       step=step, evidence=[outp[-300:]],
                       action="Stop the run by hand (box ssh may be down), e.g. " + " ".join(sc["argv"]))
            self.event_log(f"eval judge stop command failed at step {step}, rc={rc}; retrying each poll")
            self._log_md(f"**{tu.fmt12(time.time(), date=True)}: stop command FAILED** `{' '.join(sc['argv'])}` -> "
                         f"rc {rc} after {ps['attempts']} attempts; retrying every poll\n\n```\n{outp[-400:]}\n```")
        self.save()
        return False

    def _followup(self) -> str:
        fc = self.cfg["stop_command"].get("followup_check")
        base = "The trainer should save its final state and exit. "
        if not fc:
            return base + "No clean-up check configured: make sure something destroys / releases the machine."
        try:
            r = subprocess.run(fc["argv"], capture_output=True, text=True, timeout=30, stdin=subprocess.DEVNULL)
            ok = r.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            ok = False
        return base + (fc.get("ok_msg") or "The clean-up process is running.") if ok else \
            base + (fc.get("missing_msg") or "NO clean-up process is running: start it by hand.")

    def _sleep(self, secs: float):
        end = time.time() + secs
        while time.time() < end and not getattr(self, "_stop_flag", {}).get("flag"):
            time.sleep(min(1.0, max(0.0, end - time.time())))

    def _log_md(self, text: str):
        with open(os.path.join(self.out, "judge-log.md"), "a") as f:
            f.write(f"\n{text}\n")

    # -- independent confirmation before acting
    def confirm(self, v: dict, dec: dict, sets: dict) -> tuple[bool, str]:
        cc = self.cfg["policy"].get("confirm") or {}
        before, cur_step = dec.get("before_step"), (dec.get("steps") or [v["step"]])[-1]
        if before is None or before not in sets or cur_step not in sets:
            return False, f"confirmation needs the sets {before} and {cur_step}; not available"
        bpaths, cpaths = self.local_paths(before, sets[before]), self.local_paths(cur_step, sets[cur_step])
        variants = cc.get("variants") or sorted({x for _p, x in cpaths})
        sub = f"confirm-vs-{before}"
        sheets = []
        named = [p for p in (dec.get("prompts") or []) if any((p, x) in cpaths for x in variants)]
        if named:
            sheets += self.make_sheets(cur_step, cpaths, before, bpaths, variants=variants, prompts=named,
                                       tile=int(cc.get("zoom_tile") or 576),
                                       rows_per_sheet=int(cc.get("zoom_rows_per_sheet") or 2), subdir=sub + "/zoom",
                                       prev_label="BEFORE", cur_label="CUR")
        n_zoom = len(sheets)
        sheets += self.make_sheets(cur_step, cpaths, before, bpaths, variants=variants, tile=int(cc.get("tile") or 288),
                                   subdir=sub + "/all", prev_label="BEFORE", cur_label="CUR")
        prompts = self.cfg["sheet"].get("prompts") or sorted({p for p, _x in cpaths}, key=lambda x: (str(type(x)), x))
        text = "\n".join([
            f"Run: {self.cfg['name']}. Compare the CURRENT set (step {cur_step}) with the BEFORE set (step {before}): "
            f"same prompts, same seeds, so differences come from training.",
            f"{len(sheets)} sheets attached, in order." + (
                f" The first {n_zoom} show the rows of prompts {', '.join('#' + str(p) for p in named)} enlarged; the "
                f"rest show every prompt." if named else " They show every prompt."),
            "", "Prompts:", *self._prompt_lines(prompts)])
        jc = dict(self.cfg["judge"], effort=cc.get("effort") or "high", _schema=CONFIRM_SCHEMA)
        instr = confirm_instructions(jc)
        res = self._call(sheets, text, instr, jc, f"confirm {cur_step} vs {before}")
        if res is None:
            return False, "credit cap: no confirmation call possible"
        if not res.get("ok"):
            return False, f"confirmation call failed: {res.get('error', '')[:200]}"
        c = res["verdict"] if isinstance(res.get("verdict"), dict) else {}
        try:
            conf = float(c.get("confidence") or 0)
            conf = conf / 100 if conf > 1 else conf
        except (TypeError, ValueError):
            conf = 0.0
        minc = float(cc.get("min_confidence", 0.7))
        st_, lk = c.get("structure"), c.get("look")
        if dec["rule"] == "structure_regression":
            ok = st_ == "worse" and conf >= minc
        else:
            ok = st_ in ("same", "worse") and lk in ("same", "worse") and conf >= minc
        why = (f"step {cur_step} vs {before}: structure {st_}, look {lk}, confidence {conf:.2f}; "
               f"{str(c.get('reason') or '')[:300]}" + (f" Regressions: {'; '.join(map(str, (c.get('regressions') or [])[:4]))}"
                                                        if c.get("regressions") else ""))
        with open(os.path.join(self.out, "confirmations.jsonl"), "a") as f:
            f.write(json.dumps({"t": time.time(), "time": tu.fmt12(time.time(), date=True), "step": cur_step,
                                "before": before, "rule": dec["rule"], "verdict": c, "agree": ok, "sheets": sheets},
                               default=str) + "\n")
        return ok, why

    def _human_log(self, v: dict, dec: dict):
        p = os.path.join(self.out, "judge-log.md")
        new = not os.path.exists(p)
        c = self.st["credits"]
        with open(p, "a") as f:
            if new:
                f.write(f"# Eval judge log: {self.cfg['name']}\n\nModel {self.cfg['judge']['model']} "
                        f"(effort {self.cfg['judge']['effort']}). Scores 1-5 (3 = base level). Stop policy: "
                        f"{json.dumps(self.cfg['policy'])}. act = {self.cfg.get('act')}.\n")
            f.write(f"\n## Step {v['step']}" + (f" vs {v['prev_step']}" if v.get("prev_step") is not None else "")
                    + f" ({v['time']})\n\n"
                    f"- **Structure {v['structure_score']}/5**, vs previous: {v['vs_previous']['structure']}, "
                    f"vs reference: {v['vs_reference']['structure']}\n"
                    f"- **Look {v['look_score']}/5**, vs previous: {v['vs_previous']['look']}, vs reference: "
                    f"{v['vs_reference']['look']}\n"
                    f"- Previous regressions persist: {v['previous_regressions_persist']}\n"
                    f"- Model recommends **{v['recommend']}** (confidence {v['confidence']:.2f}). "
                    f"Policy decision: **{dec['decision'].upper()}**" + (f" ({dec['rule']})" if dec.get("rule") else "")
                    + f": {dec['why']}\n")
            if v.get("regressions"):
                f.write("- Regressions:\n" + "".join(f"  - {x}\n" for x in v["regressions"]))
            if v.get("notable"):
                f.write("- Notable:\n" + "".join(f"  - {x}\n" for x in v["notable"]))
            f.write(f"- Reason: {v['reason']}\n")
            f.write("- Sheets: " + ", ".join(os.path.relpath(s, self.out) for s in v["sheets"]) + "\n")
            f.write(f"- Call: {v.get('secs')} s, credits this call {v.get('credits_call')}, run total "
                    f"{c['used']:.2f} of cap {self.cfg['judge']['credit_cap']}\n")

    def precheck(self) -> bool | None:
        """Run stop_command.precheck (e.g. `ssh -F ... box true`) once: warn now if the stop path is broken."""
        pc = self.cfg["stop_command"].get("precheck")
        if not pc or not self._acting():
            return None
        try:
            r = subprocess.run(pc, cwd=self.cfg["stop_command"].get("cwd") or None, capture_output=True, text=True,
                               timeout=60, stdin=subprocess.DEVNULL)
            ok, outp = r.returncode == 0, ((r.stdout or "") + (r.stderr or ""))[-200:]
        except (OSError, subprocess.TimeoutExpired) as e:
            ok, outp = False, f"{type(e).__name__}: {e}"
        self.st["precheck"] = {"t": time.time(), "ok": ok, "output": outp}
        self.log(f"stop-path precheck: {'ok' if ok else 'FAILED'} {outp.strip()[:150]}")
        if not ok:
            self.alert("warn", "eval judge: the stop path does not work right now (precheck failed); a stop decision "
                               "would be retried every poll until it works", rule="eval_judge_precheck",
                       evidence=[outp], action="Check the box ssh (the pod may be stopped or its ssh port changed).")
        return ok

    # -- loop
    def _new_launch(self, launch: float | None):
        """The source shows a new launch (restart from scratch): archive the state of the old one, start fresh."""
        old = self.st.get("launch")
        self.st["launch"] = launch
        if old is None or launch is None or abs(launch - old) < 1 or self.st.get("acted"):
            return
        arch = os.path.join(self.out, f"state-launch-{int(old)}.json")
        with open(arch, "w") as f:
            json.dump(self.st, f, indent=1, default=str)
        for k, empty in (("judged", {}), ("failed", {}), ("history", []), ("decisions", []), ("cache", {}),
                         ("recommended_stop", []), ("confirmations", []), ("pending_stop", None)):
            self.st[k] = copy.deepcopy(empty)
        msg = (f"eval judge: the run restarted (new launch, first file at {tu.fmt12(launch, date=True)}); earlier "
               f"verdicts archived to {os.path.basename(arch)}, judging the new launch from its first set")
        self.log(msg)
        self.alert("warn", msg, rule="eval_judge_launch")
        self._log_md(f"\n---\n\n**{tu.fmt12(time.time(), date=True)}: new launch detected** (first file "
                     f"{tu.fmt12(launch, date=True)}); verdicts below are for the new launch.")
        self.event_log(msg[:300])

    def tick(self) -> list[int]:
        if self.st.get("pending_stop") and not self.st.get("acted"):
            self.try_stop(1)
            if self.st.get("acted"):
                return []
        sets, launch = self.list_sets(with_launch=True)
        self._new_launch(launch)
        # a judged set that changed afterwards (more images, or rewritten by a resume) is judged again
        for k in list(self.st["judged"]):
            sig, cur = (self.st["judged"][k] or {}).get("sig"), sets.get(int(k))
            if sig and cur and (cur["n"] > sig.get("n", 0) or cur["newest"] > sig.get("newest", 0) + 60) \
                    and not self.st.get("acted") and not self.st.get("pending_stop"):
                self.log(f"step {k}: set changed after judging ({sig} -> n {cur['n']}); judging it again")
                self.alert("info", f"eval judge: the step {k} set changed after it was judged "
                                   f"({sig.get('n')} -> {cur['n']} images); judging it again", rule="eval_judge_rejudge",
                           step=int(k))
                del self.st["judged"][k]
                self.st["history"] = [h for h in self.st["history"] if str(h["step"]) != k]
        done = {int(k) for k in self.st["judged"]} | {int(k) for k in self.st["failed"]}
        now = time.time()
        steps = ready_steps(sets, self.cfg["source"], now, done)
        judged = []
        for step in steps:
            if self.st.get("acted"):
                break
            cur, _bad = self.usable(step, sets[step], now)
            if cur is None:
                break                                      # judge in step order: later sets wait for this one
            v = self.judge_step(step, sets, cur)
            if v is None:
                if self.dry:
                    self.st["failed"][str(step)] = "dry run"
                continue
            v["sig"] = {"n": sets[step]["n"], "newest": sets[step]["newest"]}
            self.record(v, sets)
            judged.append(step)
            self.log(f"step {step}: S{v['structure_score']} L{v['look_score']} {v['vs_previous']} -> "
                     f"{v['decision']['decision']}")
        try:
            self.gates_tick()
        except Exception as e:  # noqa: BLE001 - the gate view is informational
            self.log(f"gates: {type(e).__name__}: {e}")
        self.save()
        return judged

    def list_sets(self, with_launch: bool = False):
        src = self.cfg["source"]
        files = list_ssh(src["ssh"], src["dir"], src.get("timeout_s", 120)) if src.get("ssh") else list_local(src["dir"])
        return group_sets(files, src, with_launch=with_launch)

    # -- early-stop gate outputs (informational: summary + sheets into the judge log, optional model look)
    def gates_tick(self):
        gc = self.cfg.get("gates") or {}
        if not gc.get("dir"):
            return
        src = self.cfg["source"]
        ssh = src.get("ssh") if gc.get("ssh", "source") == "source" else gc.get("ssh")
        if ssh:
            files = list_ssh(ssh, gc["dir"], src.get("timeout_s", 120))
        else:
            files = list_local(gc["dir"]) if os.path.isdir(gc["dir"]) else []
        summ, pat = gc.get("summary_file") or "GATE.json", gc.get("sheet_regex") or r"^sheet-\d+\.jpg$"
        by: dict[str, dict] = {}
        for rel, size, mt in files:
            parts = rel.split("/")
            if len(parts) == 2 and size > 0 and (parts[1] == summ or re.match(pat, parts[1])):
                by.setdefault(parts[0], {})[parts[1]] = (size, mt)
        seen = self.st.setdefault("gates", {})
        self.st["gates_seen_dirs"] = sorted(by)
        now = time.time()
        for label, fs in sorted(by.items(), key=lambda kv: (not kv[0].isdigit(), int(kv[0]) if kv[0].isdigit() else 0)):
            if label in seen or summ not in fs:
                continue
            n_sheets = sum(1 for f in fs if f != summ)
            newest = max(mt for _s, mt in fs.values())
            if now - newest < 60 or (n_sheets < int(gc.get("sheets", 2)) and now - fs[summ][1] < 1200):
                continue
            dest = os.path.join(self.out, "gates", label)
            os.makedirs(dest, exist_ok=True)
            if ssh:
                fetch_ssh(ssh, gc["dir"], [f"{label}/{f}" for f in sorted(fs)], dest)
                paths = {f: os.path.join(dest, f"{label}__{f}") for f in fs}
            else:
                paths = {f: os.path.join(gc["dir"], label, f) for f in fs}
            sheets = sorted(p for f, p in paths.items() if f != summ and image_complete(p))
            try:
                with open(paths[summ]) as f:
                    g = json.load(f)
            except (OSError, ValueError):
                continue                                   # half-copied JSON: next poll
            sm, al = g.get("summary") or {}, g.get("alerts") or []
            keys = ("n", "ft_vs_base_edge", "ft_vs_base_edge_seed1", "ref_vs_base_edge_seed1", "drift_ratio_vs_F",
                    "bad_images", "missing")
            short = {k: sm.get(k) for k in keys if k in sm}
            look = self._gate_look(label, sheets) if gc.get("judge") and sheets and not self.dry else None
            seen[label] = {"t": now, "summary": short, "alerts": al, "sheets": sheets, "look": look}
            sev = "warn" if al or (look and look.get("structure") == "worse") else "info"
            what = (f"eval judge: early-stop gate {label} outputs arrived: drift vs F {short.get('drift_ratio_vs_F')}, "
                    f"FT-vs-Base edge {short.get('ft_vs_base_edge')}" + (f"; gate ALERTS: {'; '.join(al)}" if al else
                                                                        "; no gate alerts") +
                    (f"; model look: structure {look.get('structure')} / look {look.get('look')} vs Base "
                     f"({look.get('confidence')})" if look else ""))
            self.alert(sev, what[:600], rule="eval_gate", step=label, evidence=[f"sheets: {dest}"],
                       action="Informational (the in-run sample judge drives the stop policy); open the sheets.")
            with open(os.path.join(self.out, "judge-log.md"), "a") as f:
                f.write(f"\n## Early-stop gate {label} ({tu.fmt12(now, date=True)})\n\n"
                        f"- Metrics: {json.dumps(short)}\n- Gate alerts: {'; '.join(al) or 'none'}\n"
                        + (f"- Model look (vs Base, informational): structure {look.get('structure')}, look "
                           f"{look.get('look')}, confidence {look.get('confidence')}: {look.get('reason')}\n"
                           if look else "")
                        + "- Sheets: " + ", ".join(os.path.relpath(p, self.out) if p.startswith(self.out) else p
                                                  for p in sheets) + "\n")
            self.event_log(what[:300])

    def _gate_look(self, label: str, sheets: list[str]) -> dict | None:
        gc = self.cfg.get("gates") or {}
        jc = dict(self.cfg["judge"], _schema=CONFIRM_SCHEMA, effort=gc.get("effort") or self.cfg["judge"]["effort"])
        text = (f"Early-stop gate of run {self.cfg['name']} at update {label}. {len(sheets)} sheets attached. "
                + (gc.get("layout") or "Columns: Base | a reference finetune | the current finetune at seeds 1-3.") +
                " Treat the Base column as BEFORE and the finetune columns as CURRENT: is the finetune's structure "
                "better, the same or worse than Base, and its look?")
        res = self._call(sheets, text, confirm_instructions(jc), jc, f"gate {label}")
        if not res or not res.get("ok"):
            return None
        return res["verdict"] if isinstance(res.get("verdict"), dict) else None

    def run(self, once: bool = False) -> str:
        lc = self.cfg["lifecycle"]
        t0 = time.time()
        self.st["started"] = self.st.get("started") or t0
        deadline = tu.parse_local(lc.get("deadline_local")) or (t0 + float(lc.get("max_hours", 14)) * 3600)
        deadline = min(deadline, t0 + float(lc.get("max_hours", 14)) * 3600)
        pidf = os.path.join(self.out, "watcher.pid")
        with open(pidf, "w") as f:
            f.write(str(os.getpid()))
        stop = {"flag": False}

        def _sig(*_):
            stop["flag"] = True
        signal.signal(signal.SIGTERM, _sig)
        self._stop_flag = stop
        self.precheck()
        m = self.meter_fn()
        if m and self.st["credits"]["start"] is None:
            self.st["credits"]["start"] = m.get("credits")
        self.log(f"start: {self.cfg['name']}, deadline {tu.fmt12(deadline, date=True)}, act={self.cfg.get('act')}"
                 f"{' (no-act)' if self.no_act else ''}{' DRY' if self.dry else ''}, meter {m}")
        why = "?"
        errors = 0
        try:
            while True:
                try:
                    self.tick()
                    errors = 0
                    self.status(state="watching", error=None)
                except Exception as e:  # noqa: BLE001 - keep watching through transient ssh/listing errors
                    errors += 1
                    self.log(f"tick error {errors}: {type(e).__name__}: {e}")
                    self.status(state="error", error=f"{type(e).__name__}: {e}"[:300])
                    if errors == 5:
                        self.alert("warn", f"eval judge: 5 failed polls in a row ({type(e).__name__}: {str(e)[:150]})",
                                   rule="eval_judge_error")
                last = lc.get("last_step")
                if once:
                    why = "once"
                    break
                if self.st.get("acted") and lc.get("exit_after_act", True):
                    why = "stop decision handled"
                    break
                if last and any(int(k) >= int(last) for k in self.st["judged"]):
                    why = f"last step {last} judged"
                    break
                if time.time() >= deadline:
                    why = "deadline"
                    break
                if stop["flag"]:
                    why = "SIGTERM"
                    break
                for _ in range(int(self.cfg.get("poll_s", 120))):
                    if stop["flag"]:
                        break
                    time.sleep(1)
        finally:
            c = self.st["credits"]
            msg = (f"eval judge exiting ({why}): judged steps {sorted(int(k) for k in self.st['judged'])}, "
                   f"credits used {c['used']:.2f} in {c['calls']} calls")
            self.log(msg)
            self.save()
            self.status(state="exited", exited=True, why=why)
            if not once:
                # exit record only in our own alerts file: the watcher's `tail` treats rule watcher_exit as its end
                rec = {"t": time.time(), "time": tu.fmt12(time.time(), date=True), "status": "new", "severity": "info",
                       "rule": "watcher_exit", "key": "eval_judge_exit", "what": msg, "source": "eval_judge"}
                with open(os.path.join(self.out, "alerts.jsonl"), "a") as f:
                    f.write(json.dumps(rec) + "\n")
                self.event_log(msg)
            try:
                os.remove(pidf)
            except OSError:
                pass
        return msg
