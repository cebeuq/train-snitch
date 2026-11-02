#!/usr/bin/env python3
"""READ-ONLY file reader used by every file-based source, locally or on a remote host.

Self-contained and stdlib-only, so the same file runs in two ways:
  local:   python3 -I reader.py '<request json>'
  remote:  ssh HOST python3 -I - '<request json>'   (this file is sent on stdin)

Request: {"ops": [{"id": ..., "op": ..., ...}], "offsets": {key: byte offset}}
Reply (one JSON line): {"results": {id: ...}, "offsets": {key: offset}, "now": epoch}

Ops (each opens files read-only and writes nothing):
  tail      {glob, all?, first_tail?, max_bytes?}  new complete lines since the stored offset (newest match, or all)
  csv       {glob, first_tail?}                    like tail, plus the header line
  tfevents  {glob, tags?}                          new scalar points from TensorBoard event files
  wandb     {path, first_tail?}                    newest W&B run dir: wandb-summary.json + new output.log lines
  stat      {glob, limit?}                         [{path, size, mtime}] newest first
  disk      {path}                                 free / total GB of the filesystem holding path
  json      {path, max_bytes?}                     parsed JSON file (or null)
  exists    {path}                                 true / false
"""
import glob as _glob
import json
import os
import shutil
import struct
import sys
import time

MAXB = 400_000


def _expand(p):
    return os.path.expanduser(os.path.expandvars(p))


def _matches(pattern):
    ms = [m for m in _glob.glob(_expand(pattern), recursive=True) if os.path.isfile(m)]
    ms.sort(key=lambda m: (os.path.getmtime(m), m))
    return ms


def _new_lines(path, key, offs, out_offs, first_tail=50, max_bytes=MAXB):
    try:
        size = os.path.getsize(path)
    except OSError:
        return None
    off = offs.get(key)
    with open(path, "rb") as f:
        if off is None or off > size:          # first read (or file was truncated / rotated): last lines only
            f.seek(max(0, size - max_bytes))
            lines = f.read().decode("utf-8", "replace").splitlines()[-first_tail:] if first_tail else []
            out_offs[key] = size
            return lines
        f.seek(off)
        chunk = f.read(max_bytes)
    end = chunk.rfind(b"\n")
    if end < 0:
        out_offs[key] = off
        return []
    out_offs[key] = off + end + 1
    return chunk[: end + 1].decode("utf-8", "replace").splitlines()


def op_tail(o, offs, oo):
    ms = _matches(o["glob"])
    if not ms:
        return {"files": {}, "missing": True}
    pick = ms if o.get("all") else ms[-1:]
    res = {}
    for m in pick[-20:]:
        x = _new_lines(m, o["id"] + ":" + m, offs, oo, int(o.get("first_tail", 50)), int(o.get("max_bytes", MAXB)))
        if x is not None:
            res[m] = x
    return {"files": res, "missing": False}


def op_csv(o, offs, oo):
    r = op_tail(o, offs, oo)
    for m in list(r["files"]):
        try:
            with open(m, encoding="utf-8", errors="replace") as f:
                hdr = f.readline().rstrip("\n")
        except OSError:
            hdr = ""
        r["files"][m] = {"header": hdr, "lines": [x for x in r["files"][m] if x != hdr]}
    return r


# ---- TensorBoard event files (TFRecord framing + the few Event / Summary proto fields we need) ----------
def _varint(b, i):
    x = s = 0
    while True:
        c = b[i]
        i += 1
        x |= (c & 0x7F) << s
        if not c & 0x80:
            return x, i
        s += 7


def _fields(b):
    i, n = 0, len(b)
    while i < n:
        key, i = _varint(b, i)
        fn, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(b, i)
        elif wt == 1:
            v, i = b[i:i + 8], i + 8
        elif wt == 2:
            ln, i = _varint(b, i)
            v, i = b[i:i + ln], i + ln
        elif wt == 5:
            v, i = b[i:i + 4], i + 4
        else:
            return
        yield fn, wt, v


def _tensor_scalar(b):
    dtype, vals = None, []
    for fn, wt, v in _fields(b):
        if fn == 1 and wt == 0:
            dtype = v
        elif fn == 4 and wt == 2 and v:                       # tensor_content
            vals.append(struct.unpack("<d", v[:8])[0] if dtype == 2 else struct.unpack("<f", v[:4])[0])
        elif fn == 5:                                         # float_val (packed or not)
            vals.append(struct.unpack("<f", v[:4])[0])
        elif fn == 6:                                         # double_val
            vals.append(struct.unpack("<d", v[:8])[0])
    return vals[0] if vals else None


def _event(b):
    wall, step, pts = None, 0, []
    for fn, wt, v in _fields(b):
        if fn == 1 and wt == 1:
            wall = struct.unpack("<d", v)[0]
        elif fn == 2 and wt == 0:
            step = v - (1 << 64) if v >= 1 << 63 else v
        elif fn == 5 and wt == 2:
            for fn2, wt2, val in _fields(v):
                if fn2 != 1 or wt2 != 2:
                    continue
                tag, x = None, None
                for fn3, wt3, w in _fields(val):
                    if fn3 == 1 and wt3 == 2:
                        tag = w.decode("utf-8", "replace")
                    elif fn3 == 2 and wt3 == 5:
                        x = struct.unpack("<f", w)[0]
                    elif fn3 == 8 and wt3 == 2 and x is None:
                        x = _tensor_scalar(w)
                if tag is not None and x is not None:
                    pts.append((tag, x))
    return wall, step, pts


def op_tfevents(o, offs, oo):
    tags = set(o.get("tags") or [])
    out = []
    for m in _matches(o["glob"])[-20:]:
        key = o["id"] + ":" + m
        off = int(offs.get(key, 0))
        try:
            size = os.path.getsize(m)
        except OSError:
            continue
        if off > size:
            off = 0
        with open(m, "rb") as f:
            f.seek(off)
            buf = f.read(min(size - off, 8 * MAXB))
        i = 0
        while i + 12 <= len(buf):
            (ln,) = struct.unpack("<Q", buf[i:i + 8])
            if i + 12 + ln + 4 > len(buf):
                break                                         # partial record: read it next time
            try:
                wall, step, pts = _event(buf[i + 12:i + 12 + ln])
            except (IndexError, struct.error):
                pts, wall, step = [], None, 0
            for tag, x in pts:
                if not tags or tag in tags:
                    out.append({"file": m, "wall": wall, "step": step, "tag": tag, "value": x})
            i += 12 + ln + 4
        oo[key] = off + i
    return {"points": out[-5000:]}


def op_wandb(o, offs, oo):
    p = _expand(o["path"])
    run = None
    if os.path.isdir(os.path.join(p, "files")):
        run = p
    else:
        cands = [d for d in _glob.glob(os.path.join(p, "*run-*")) if os.path.isdir(d)]
        latest = os.path.join(p, "latest-run")
        if os.path.isdir(latest):
            run = os.path.realpath(latest)
        elif cands:
            run = max(cands, key=os.path.getmtime)
    if not run:
        return {"run": None}
    summ = None
    try:
        with open(os.path.join(run, "files", "wandb-summary.json")) as f:
            summ = json.load(f)
    except (OSError, ValueError):
        pass
    lines = _new_lines(os.path.join(run, "files", "output.log"), o["id"] + ":" + run + ":output.log", offs, oo,
                       int(o.get("first_tail", 50)))
    return {"run": os.path.basename(run), "summary": summ, "lines": lines or []}


def op_stat(o, offs, oo):
    ms = [m for m in _glob.glob(_expand(o["glob"]), recursive=True)]
    rows = []
    for m in ms:
        try:
            st = os.stat(m)
        except OSError:
            continue
        rows.append({"path": m, "size": st.st_size if os.path.isfile(m) else None, "mtime": st.st_mtime,
                     "dir": os.path.isdir(m)})
    rows.sort(key=lambda r: r["mtime"], reverse=True)
    return {"items": rows[: int(o.get("limit", 200))], "count": len(rows)}


def op_disk(o, offs, oo):
    try:
        u = shutil.disk_usage(_expand(o["path"]))
    except OSError as e:
        return {"error": str(e)}
    return {"free_gb": round(u.free / 1e9, 2), "total_gb": round(u.total / 1e9, 2)}


def op_json(o, offs, oo):
    try:
        with open(_expand(o["path"]), "rb") as f:
            return json.loads(f.read(int(o.get("max_bytes", MAXB))).decode("utf-8", "replace"))
    except (OSError, ValueError):
        return None


def op_exists(o, offs, oo):
    return os.path.exists(_expand(o["path"]))


OPS = {"tail": op_tail, "csv": op_csv, "tfevents": op_tfevents, "wandb": op_wandb, "stat": op_stat,
       "disk": op_disk, "json": op_json, "exists": op_exists}


def handle(req):
    offs, oo, res = req.get("offsets") or {}, {}, {}
    for o in req.get("ops") or []:
        try:
            res[o["id"]] = OPS[o["op"]](o, offs, oo)
        except Exception as e:                                # one bad op never hides the others
            res[o["id"]] = {"error": repr(e)[:300]}
    return {"results": res, "offsets": oo, "now": time.time()}


if __name__ == "__main__":
    print(json.dumps(handle(json.loads(sys.argv[1]))))
