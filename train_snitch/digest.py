"""The compact text the judge reads each tick: run facts, rule-layer state, and what changed since its last look.

Kept small on purpose (a few KB): the judge runs every minute, so every token is paid ~60x per hour.
"""
from __future__ import annotations

import json
import math
import statistics

from .engine import SEV
from .timeutil import fmt12, mins


def _f(x, nd=4):
    if isinstance(x, float):
        if math.isnan(x):
            return "NaN"
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        return round(x, nd)
    return x


def build(t: float, view: dict, hist, recs: list, active: list, summary: dict, since_t: float, since_step,
          deadline: float | None = None, cycle: int = 0) -> tuple[str, bool]:
    """Returns (digest text, changed?) where changed = new rows / new notable log lines / new records."""
    L = [f"Run: {json.dumps({k: v for k, v in (summary or {}).items() if v not in (None, '', [])}, default=str)[:600]}",
         f"Now: {fmt12(t, date=True)} (watcher cycle {cycle}" + (f", exits by {fmt12(deadline)}" if deadline else "") + ")"]
    changed = False
    gap = f"{mins(t - since_t)} min" if since_t else "start"
    # ---- rule layer
    act = sorted(active, key=lambda a: -SEV[a["severity"]])
    L.append("Active rule alerts: " + ("; ".join(f"[{a['severity']}] {a['key']}: {a['what']} (since {fmt12(a['since'])}, x{a['count']})"
                                                 for a in act[:8]) if act else "none"))
    other = [r for r in recs if r.get("source", "rules") == "rules" and r["status"] in ("resolved", "ongoing")]
    if other:
        L.append("Rule records this cycle: " + "; ".join(f"{r['status']} {r['key']}" for r in other[:6]))
    if recs:
        changed = True
    # ---- training progress since the judge's last look
    L.append(f"Since your last look ({gap}):")
    losses = [(s, x) for s, x in hist.losses if since_step is None or (s is not None and s > since_step)]
    newrows = [r for r in hist.new_train_rows]
    if losses or newrows:
        changed = True
        fin = [x for _, x in losses if isinstance(x, float) and math.isfinite(x)]
        bad = len(losses) - len(fin)
        st = [s for s, _ in losses if s is not None]
        L.append(f"  train rows: {len(losses)} (updates {min(st) if st else '?'} -> {max(st) if st else '?'})"
                 + (f", loss last {_f(fin[-1])} min {_f(min(fin))} max {_f(max(fin))} mean {_f(statistics.fmean(fin))}" if fin else "")
                 + (f", NON-FINITE {bad}" if bad else ""))
    else:
        L.append(f"  train rows: none new (last update {hist.last_train_step})")
    tail = [_f(x[1]) for x in list(hist.losses)[-16:]]
    if tail:
        L.append(f"  loss trend (last {len(tail)} rows): {tail}")
    for kind, ser in hist.val.items():
        s = list(ser)[-5:]
        if s:
            L.append(f"  val {kind}: " + ", ".join(f"u{u}={_f(x, 5)}" for u, x in s)
                     + (f" (step-0 baseline {_f(hist.val_baseline[kind], 5)})" if kind in hist.val_baseline else ""))
    pr = view.get("progress") or {}
    if pr:
        lr = pr.get("last_row_t")
        L.append(f"  progress: update {pr.get('step')}, last row {fmt12(lr) if lr else 'never'}"
                 + (f" ({mins((pr.get('now') or t) - lr)} min ago)" if lr else "")
                 + f", trainer alive {pr.get('alive')}, done {pr.get('done')}")
    if view.get("state_line"):
        L.append(f"  run state line: {view['state_line'][:200]}")
    fl = view.get("flags")
    if fl:
        L.append("  status flags: " + ", ".join(f"{k}={str(v)[:60]}" for k, v in fl.items()))
    # ---- notable new log lines (error-looking, benign noise removed by the engine)
    errs = [f"{src}: {ln}" for (tt, src, ln) in hist.error_lines if tt > (since_t or 0) - 1]
    if errs:
        changed = True
        L.append(f"  notable new log lines ({len(errs)}):")
        L += [f"    {e[:240]}" for e in errs[-12:]]
    nl = {k: len(v or []) for k, v in (view.get("log_lines") or {}).items() if v}
    if nl:
        L.append("  new log lines per source: " + ", ".join(f"{k} {n}" for k, n in list(nl.items())[:12]))
    # ---- hardware / money / transport
    g = view.get("gpu_util_now") or {k: v[-1][1] for k, v in hist.gpu.items() if v}
    if g:
        L.append("  GPU util %: " + ", ".join(f"{k}:{_f(u, 0)}" for k, u in sorted(g.items())[:16]))
    if view.get("disk_free_gb") is not None:
        L.append(f"  disk free: {view['disk_free_gb']:.0f} GB")
    if view.get("credit_usd") is not None:
        L.append(f"  provider credit: ${view['credit_usd']:.2f}" + (f", planned rate ${view['planned_rate_usd_h']:.2f}/h"
                                                                   if view.get("planned_rate_usd_h") else ""))
    for i in view.get("instances") or []:
        L.append(f"  instance {i.get('id')}: {i.get('actual_status')}, ${_f(float(i.get('dph_total') or 0), 2)}/h, gpu_util {i.get('gpu_util')}")
    if view.get("instances") == []:
        L.append("  instance: none (destroyed or not created)")
    if view.get("ssh") is not None:
        s = view["ssh"]
        L.append("  ssh to box: " + ("ok" if s.get("ok") else f"FAILING ({(s.get('stderr_tail') or ['?'])[-1][:120]})"))
    for k, p in (view.get("processes") or {}).items():
        L.append(f"  process {k}: {'alive' if p.get('alive') else 'NOT RUNNING'}")
    for k, c in (view.get("checkpoints") or {}).items():
        nm = c.get("newest_mtime")
        L.append(f"  checkpoints {k}: {c.get('count')} total, newest {mins((c.get('now') or t) - nm)} min old" if nm else
                 f"  checkpoints {k}: none yet")
    for k, e in (view.get("source_errors") or {}).items():
        L.append(f"  source {k} unreadable: {str((e or ['?'])[-1])[:140]}")
    if view.get("hub_procs") is not None:
        L.append("  run helper processes: " + (", ".join(sorted(view["hub_procs"])) or "none"))
    return "\n".join(L), changed
