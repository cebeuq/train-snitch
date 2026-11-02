"""Parsers that turn what common training stacks print into metric rows {step, loss, kind, extra}.

Text-log formats (`format` of a `log` source):
  hf         Hugging Face Trainer / TRL:  {'loss': 0.53, 'grad_norm': 1.2, 'learning_rate': 2e-05, 'epoch': 0.4}
             (eval lines with 'eval_loss' become val rows). No step in the line: the row counter is used.
  lightning  PyTorch Lightning progress bar:  Epoch 0:  45%|###  | 450/1000 [01:02<01:10, loss=0.123, v_num=0]
  deepspeed  DeepSpeed / Megatron:  [Rank 0] step=100, skipped=0, lr=[...] ...  |  iteration 100/ 2000 | ... lm loss: 2.1E+00 |
  keyvalue   generic accelerate / torchrun / custom prints:  step 100 ... loss=0.53  |  it: 100, loss: 0.53
  auto       try hf, then deepspeed, then lightning, then keyvalue (default)
  regex      your own: {"regex": "...(?P<step>\\d+)...(?P<loss>[-\\d.eE+naif]+)"}
Plain failure lines are matched separately by the rule layer's log patterns.
"""
from __future__ import annotations

import ast
import math
import re

NUM = r"[-+]?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?|nan|NaN|inf|Inf|-inf)"


def num(x):
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().strip(",")
    try:
        return float(s)
    except ValueError:
        low = s.lower()
        if low in ("nan", "+nan", "-nan"):
            return math.nan
        if low in ("inf", "+inf", "infinity"):
            return math.inf
        if low in ("-inf", "-infinity"):
            return -math.inf
        try:  # torch prints tensor(0.53, device='cuda:0')
            m = re.search(r"tensor\((" + NUM + ")", s)
            return float(m.group(1)) if m else None
        except ValueError:
            return None


_HF = re.compile(r"\{\s*'(?:loss|eval_loss|train_loss)'.*\}")
_LIGHTNING = re.compile(r"(?:Epoch\s+(?P<epoch>\d+)).*?\|\s*(?P<step>\d+)/(?P<total>\d+|\?)\s*\[.*?(?:train_)?loss(?:_step)?=(?P<loss>" + NUM + ")")
_DS_ITER = re.compile(r"iteration\s+(?P<step>\d+)\s*/\s*\d+.*?(?:lm )?loss:\s*(?P<loss>" + NUM + ")", re.I)
_DS_STEP = re.compile(r"\bstep=(?P<step>\d+),\s*skipped=(?P<skipped>\d+).*?(?:loss=(?P<loss>" + NUM + "))?")
_KV_STEP = re.compile(r"\b(?:global_step|step|iter(?:ation)?|it|update)s?\b\s*[=:]?\s*\[?(?P<step>\d+)", re.I)
_KV_LOSS = re.compile(r"\b(?:train[_/ ])?loss\b\s*[=:]\s*(?P<loss>" + NUM + r"|tensor\([^)]*\))", re.I)
_KV_VAL = re.compile(r"\b(?:val(?:id(?:ation)?)?|eval)[_/ ]loss\b\s*[=:]\s*(?P<loss>" + NUM + ")", re.I)


def _hf(line):
    m = _HF.search(line)
    if not m:
        return None
    s = re.sub(r"(?<!['\w])(-?)(nan|inf)\b", lambda x: f"'{x.group(1)}{x.group(2)}'", m.group(0))
    try:
        d = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return None
    if not isinstance(d, dict):
        return None
    if "eval_loss" in d:
        return {"kind": "val", "loss": num(d["eval_loss"]), "extra": {k: d[k] for k in list(d)[:8]}}
    if "loss" in d:
        ex = {k: num(d[k]) for k in ("grad_norm", "learning_rate", "epoch") if k in d}
        return {"kind": "train", "loss": num(d["loss"]), "extra": ex}
    return None


def _lightning(line):
    m = _LIGHTNING.search(line.split("\r")[-1])
    if not m:
        return None
    return {"kind": "train", "loss": num(m.group("loss")), "step": int(m.group("step")),
            "extra": {"epoch": int(m.group("epoch"))}}


def _deepspeed(line):
    m = _DS_ITER.search(line)
    if m:
        return {"kind": "train", "loss": num(m.group("loss")), "step": int(m.group("step")), "extra": {}}
    m = _DS_STEP.search(line)
    if m and m.group("loss"):
        return {"kind": "train", "loss": num(m.group("loss")), "step": int(m.group("step")),
                "extra": {"skipped": int(m.group("skipped"))}}
    return None


def _keyvalue(line):
    mv = _KV_VAL.search(line)
    if mv:
        ms = _KV_STEP.search(line)
        return {"kind": "val", "loss": num(mv.group("loss")), "step": int(ms.group("step")) if ms else None, "extra": {}}
    ml = _KV_LOSS.search(line)
    if not ml:
        return None
    ms = _KV_STEP.search(line)
    return {"kind": "train", "loss": num(ml.group("loss")), "step": int(ms.group("step")) if ms else None, "extra": {}}


FORMATS = {"hf": [_hf], "lightning": [_lightning], "deepspeed": [_deepspeed], "keyvalue": [_keyvalue],
           "auto": [_hf, _deepspeed, _lightning, _keyvalue]}


def log_rows(lines, fmt="auto", regex=None, counter=None):
    """Metric rows from text log lines. `counter` = {"n": rows so far} numbers rows that carry no step."""
    counter = counter if counter is not None else {"n": 0}
    if fmt == "none":
        return []
    rx = re.compile(regex) if regex else None
    fns = FORMATS.get(fmt, FORMATS["auto"])
    out = []
    for ln in lines or []:
        r = None
        if rx:
            m = rx.search(ln)
            if m:
                g = m.groupdict()
                r = {"kind": g.get("kind") or "train", "loss": num(g.get("loss")),
                     "step": int(g["step"]) if g.get("step") else None, "extra": {}}
        else:
            for fn in fns:
                r = fn(ln)
                if r:
                    break
        if not r or r.get("loss") is None:
            continue
        counter["n"] += 1
        if r.get("step") is None:
            r["step"] = counter["n"]
        r["raw"] = ln[:300]
        out.append(r)
    return out


def dict_row(d: dict, step_key="step", loss_key="loss", kind=None, kind_key=None, val_loss_key=None):
    """One metric row from a JSON/CSV/W&B dict, or None."""
    if not isinstance(d, dict):
        return None
    step = d.get(step_key)
    try:
        step = int(float(step)) if step not in (None, "") else None
    except (TypeError, ValueError):
        step = None
    k = (d.get(kind_key) if kind_key else None) or kind or "train"
    loss = num(d.get(loss_key))
    if loss is None and val_loss_key and d.get(val_loss_key) is not None:
        k, loss = "val", num(d.get(val_loss_key))
    if loss is None:
        return None
    return {"step": step, "loss": loss, "kind": k, "t": num(d.get("_timestamp") or d.get("time") or d.get("timestamp")),
            "extra": {kk: d[kk] for kk in list(d)[:10] if isinstance(d[kk], (int, float, str)) and kk not in (step_key, loss_key)}}


def csv_rows(header: str, lines: list, **kw):
    import csv
    cols = next(csv.reader([header])) if header else []
    out = []
    for vals in csv.reader(lines or []):
        if len(vals) != len(cols):
            continue
        r = dict_row(dict(zip(cols, vals)), **kw)
        if r:
            out.append(r)
    return out


# ---- GPU tools ------------------------------------------------------------------------------------------
NVIDIA_QUERY = "index,utilization.gpu,power.draw,memory.used,memory.total,temperature.gpu"


def nvidia_rows(lines, t):
    """`nvidia-smi --query-gpu=index,utilization.gpu,power.draw,memory.used,memory.total,temperature.gpu
    --format=csv,noheader,nounits` -> [[t, idx, util, power_w, raw, mem_used_mib, mem_total_mib]]"""
    out = []
    for ln in lines or []:
        p = [x.strip() for x in ln.split(",")]
        if len(p) < 3:
            continue
        try:
            idx, util = int(p[0]), float(p[1].split()[0])
        except (ValueError, IndexError):
            continue
        pw = num(p[2].split()[0]) if p[2] else None
        mu = num(p[3].split()[0]) if len(p) > 3 else None
        mt = num(p[4].split()[0]) if len(p) > 4 else None
        out.append([t, idx, util, pw, ln.strip(), mu, mt])
    return out


def rocm_rows(data, t):
    """`rocm-smi --showuse --showpower --showmemuse --json` -> same shape as nvidia_rows."""
    out = []
    if not isinstance(data, dict):
        return out
    for card, d in sorted(data.items()):
        m = re.search(r"(\d+)$", card)
        if not m or not isinstance(d, dict):
            continue
        util = pw = None
        for k, v in d.items():
            if re.search(r"GPU use", k, re.I):
                util = num(str(v).rstrip("%"))
            elif re.search(r"Power", k, re.I) and pw is None:
                pw = num(str(v).split()[0])
        if util is not None:
            out.append([t, int(m.group(1)), util, pw, f"{card}: use {util}% power {pw} W", None, None])
    return out
