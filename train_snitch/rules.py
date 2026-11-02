"""Built-in deterministic rules. Each rule: rule(t, view, hist, cfg) -> list[Finding] | None.

`view` is the adapter-normalized snapshot (see adapters/base.py for the fields). Return None when the rule's
input was not readable this cycle (the engine then holds the rule's active alerts instead of resolving them).
Thresholds live in cfg["rules"][<rule name>]; every rule can be disabled with {"enabled": false}.
"""
from __future__ import annotations

import math
import os
import re
import statistics

from .engine import Finding, History, SEV
from .timeutil import fmt12, parse_local, mins


def P(cfg: dict, name: str) -> dict:
    return (cfg.get("rules") or {}).get(name) or {}


def tmpl(s: str, **kw) -> str:
    class D(dict):
        def __missing__(self, k):
            return "{" + k + "}"
    try:
        return (s or "").format_map(D(**{**_vars_cfg, **kw}))
    except Exception:
        return s or ""


_vars_cfg: dict = {}


def set_vars(v: dict) -> None:
    _vars_cfg.clear()
    _vars_cfg.update(v or {})


def recent_errors(h: History, t: float, window_s: float = 1800, n: int = 6) -> list[str]:
    return [f"{src}: {ln}" for (tt, src, ln) in list(h.error_lines) if t - tt <= window_s][-n:]


def _finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


# ---------------------------------------------------------------------------------------------- flags
def flags(t, v, h, cfg):
    """Status flags written by the run (e.g. BOOT-FAILED, GATE-ALERT, SMOKE-PASS)."""
    fl = v.get("flags")
    if fl is None:
        return None
    spec_all = cfg.get("flags") or {}
    out = []
    for name, value in fl.items():
        spec = spec_all.get(name)
        if spec is None:
            if not re.search(r"FAIL", name):
                continue
            spec = {"severity": "critical", "what": "{flag}: {value}", "action": "Read the failing job's log."}
        if spec.get("ignore"):
            continue
        sval = str(value)
        m = re.match(r"\s*([A-Za-z][\w.-]*)", sval)
        phase = m.group(1) if m else sval
        sm = re.search(r'"step"\s*:\s*(\d+)', sval) or re.search(r"update (\d+)", sval)
        step = sm.group(1) if sm else "?"
        kw = dict(flag=name, value=sval[:200], phase=phase, step=step)
        ev = [f"{name}: {sval[:300]}"]
        if v.get("state_line"):
            ev.append(f"state: {v['state_line']}")
        if SEV[spec.get("severity", "info")] >= SEV["warn"]:
            ev += recent_errors(h, t)
        out.append(Finding(f"flag:{name}", spec.get("severity", "info"), tmpl(spec.get("what", "{flag}: {value}"), **kw),
                           ev, tmpl(spec.get("action", ""), **kw), kind=spec.get("kind", "condition"),
                           escalate_to_judge=bool(spec.get("judge"))))
    return out


# ---------------------------------------------------------------------------------------------- logs
_compiled: dict = {}


def _rx(p):
    r = _compiled.get(p)
    if r is None:
        r = _compiled[p] = re.compile(p)
    return r


def log_patterns(t, v, h, cfg):
    """Regex rules over new log lines (and the run's one-line state). Events, deduped by rule id."""
    pats = list(cfg.get("log_patterns") or [])
    benign = [_rx(p) for p in cfg.get("benign_patterns") or []]
    sources = dict(v.get("log_lines") or {})
    if v.get("state_line") and v.get("state_line_new", True):
        sources["state"] = [v["state_line"]]
    found: dict[str, Finding] = {}
    counts: dict[str, int] = {}
    mins_: dict[str, int] = {}
    for p in pats:
        rx = _rx(p["pattern"])
        ign = [_rx(x) for x in p.get("ignore", [])]
        src_rx = _rx(p["sources"]) if p.get("sources") else None
        for src, lines in sources.items():
            if src_rx and not src_rx.search(src):
                continue
            lines = lines or []
            for i, ln in enumerate(lines):
                m = rx.search(ln)
                if not m or any(r.search(ln) for r in ign) or any(r.search(ln) for r in benign):
                    continue
                g = {k: (val or "") for k, val in m.groupdict().items()}
                rule, sev, what, action = p["id"], p.get("severity", "warn"), p.get("what", "{line}"), p.get("action", "")
                pg = p.get("protected_group")
                if pg:
                    name = g.get(pg, "").lstrip("/")
                    if name and name in (v.get("protected") or []):
                        rule, sev = p.get("protected_rule", rule + "_protected"), p.get("protected_severity", "critical")
                        what, action = p.get("protected_what", what), p.get("protected_action", action)
                counts[rule] = counts.get(rule, 0) + 1
                mins_[rule] = int(p.get("min_count", 1))
                ctx = lines if len(lines) <= 12 else lines[max(0, i - 4): i + 5]
                ev = [f"{src}: {x[:400]}" for x in ctx]
                f = found.get(rule)
                if f is None:
                    found[rule] = Finding(rule, sev, tmpl(what, line=ln[:300], source=src, **g), ev,
                                          tmpl(action, line=ln[:300], source=src, **g), kind=p.get("kind", "event"),
                                          escalate_to_judge=bool(p.get("judge")))
                else:
                    f.evidence = f.evidence + [e for e in ev if e not in f.evidence]
                    if SEV[sev] > SEV[f.severity]:
                        f.severity = sev
    for f in found.values():
        f.evidence = f.evidence[:12]
        if counts.get(f.rule, 0) > 1:
            f.what = f"{f.what} (x{counts[f.rule]} this cycle)"
    return [f for f in found.values() if counts.get(f.rule, 0) >= mins_.get(f.rule, 1)]


# ---------------------------------------------------------------------------------------------- loss
def loss_nonfinite(t, v, h, cfg):
    rows = h.new_train_rows
    if not rows:
        return None
    bad = [r for r in rows if r.get("loss") is not None and not _finite(r.get("loss"))]
    if not bad:
        return []
    first = bad[0]
    kind = "NaN" if isinstance(first["loss"], float) and math.isnan(first["loss"]) else "inf"
    ev = [f"update {r.get('step')}: loss {r.get('loss')}" for r in bad[:6]]
    return [Finding("loss_nonfinite", "critical",
                    f"Training loss is {kind} from update {first.get('step')} ({len(bad)} of {len(rows)} new rows)",
                    ev, tmpl(P(cfg, "loss_nonfinite").get("action",
                        "Training has diverged and the box keeps billing. Check lr / grad norm in the last finite rows; "
                        "stop and resume from the last finite save (main agent decides)."))) ]


def loss_spike(t, v, h, cfg):
    p = P(cfg, "loss_spike")
    rows = [r for r in h.new_train_rows if _finite(r.get("loss"))]
    if not rows:
        return None
    prior = [x[1] for x in list(h.losses)[-int(p.get("window", 50)):]]
    if len(prior) < int(p.get("min_prior", 10)):
        return []
    med = statistics.median(prior)
    k = float(p.get("factor", 3.0))
    spikes = [r for r in rows if med > 0 and r["loss"] >= k * med]
    if len(spikes) < int(p.get("min_rows", 2)):
        return []
    peak = max(r["loss"] for r in spikes)
    ev = [f"update {r.get('step')}: loss {r['loss']:.4f}" for r in spikes[:6]] + [f"rolling median of the previous {len(prior)} rows: {med:.4f}"]
    return [Finding("loss_spike", "warn", f"Loss spike: {len(spikes)} rows at up to {peak / med:.1f}x the rolling median",
                    ev, p.get("action", "Check whether it recovers within ~50 updates; if not, compare against the last save."),
                    metric=peak / med, escalate_to_judge=True)]


# ---------------------------------------------------------------------------------------------- progress
def train_stalled(t, v, h, cfg):
    p = P(cfg, "train_stalled")
    pr = v.get("progress")
    if not pr:
        return None
    if pr.get("done") or not pr.get("started_t"):
        return []
    now = pr.get("now") or t
    started, last = pr["started_t"], pr.get("last_row_t") or 0
    warn_s, crit_s = float(p.get("warn_min", 15)) * 60, float(p.get("critical_min", 30)) * 60
    if last < started:   # no row since this launch yet: model load + caches can take long
        ref = started + float(p.get("first_row_grace_min", 60)) * 60
    else:
        ref = max(last, started + float(p.get("start_grace_min", 10)) * 60)
    idle = now - ref
    if idle <= warn_s:
        return []
    age = mins(now - max(last, started))
    ev = [f"last row at {fmt12(last) if last else 'never'}, update {pr.get('step')}, trainer alive: {pr.get('alive')}"]
    if v.get("gpu_util_now"):
        ev.append("gpu util %: " + ", ".join(f"GPU {k} {u}" for k, u in sorted(v["gpu_util_now"].items())))
    if v.get("state_line"):
        ev.append(f"state: {v['state_line']}")
    if idle > crit_s:
        return [Finding("train_stalled", "critical",
                        f"No new training row for {age} min (update {pr.get('step')}); past {int(crit_s // 60)} min",
                        ev, tmpl(p.get("critical_action", "Check the trainer log; the run's own watchdog (if any) "
                                       "should stop + auto-resume now.")))]
    return [Finding("train_stalled", "warn", f"No new training row for {age} min (update {pr.get('step')})", ev,
                    tmpl(p.get("action", "Check the trainer log / GPU use; saves and val passes pause rows only 1-2 min.")))]


def progress_reset(t, v, h, cfg):
    pr = v.get("progress")
    if not pr or pr.get("step") is None:
        return None
    step, prev = pr["step"], h.last_step
    h.last_step = step
    if prev is not None and step < prev:
        return [Finding("progress_reset", "info", f"Update counter went back from {prev} to {step} (new launch, resume or fallback)",
                        [f"state: {v.get('state_line') or ''}"], kind="event")]
    return []


# ---------------------------------------------------------------------------------------------- validation
def val_trend(t, v, h, cfg):
    p = P(cfg, "val_trend")
    if not h.new_val_rows:
        return None
    out = []
    n, tot = int(p.get("rising_points", 3)), float(p.get("rising_total", 0.02))
    for kind, ser in h.val.items():
        s = list(ser)
        if len(s) >= n + 1:
            w = s[-(n + 1):]
            if all(w[i + 1][1] > w[i][1] for i in range(n)) and w[-1][1] - w[0][1] > tot:
                out.append(Finding("val_loss_rising", "warn",
                                   f"{kind} val/loss rose {n} times in a row: {w[0][1]:.4f} -> {w[-1][1]:.4f}",
                                   [f"update {u}: val/loss {x:.5f}" for u, x in w], p.get("action",
                                       "Possible overfitting: compare samples at the last saves; early stop is a human call."),
                                   key=f"val_loss_rising:{kind}", escalate_to_judge=True))
        base = h.val_baseline.get(kind)
        if base is not None and s and s[-1][1] > base and s[-1][0] not in (0, None):
            out.append(Finding("val_above_baseline", "critical",
                               f"{kind} val/loss {s[-1][1]:.4f} at update {s[-1][0]} is above the step-0 baseline {base:.4f}",
                               [f"update {u}: val/loss {x:.5f}" for u, x in s[-4:]], p.get("baseline_action",
                                   "Training now makes validation worse than the untouched model: review and consider an early stop."),
                               key=f"val_above_baseline:{kind}"))
    return out


# ---------------------------------------------------------------------------------------------- hardware
def disk_low(t, v, h, cfg):
    p = P(cfg, "disk_low")
    d = v.get("disk_free_gb")
    if d is None:
        return None
    w, c = float(p.get("warn_gb", 50)), float(p.get("critical_gb", 15))
    if d >= w:
        return []
    sev = "critical" if d < c else "warn"
    return [Finding("disk_low", sev, f"Disk free is {d:.0f} GB (warn < {w:.0f}, critical < {c:.0f})",
                    [f"disk_free_gb {d}"], tmpl(p.get("action", "Make sure saves are being pulled off the box; free space before the next save.")))]


def gpu_util_drop(t, v, h, cfg):
    p = P(cfg, "gpu_util_drop")
    if not v.get("gpu_samples"):
        return None
    low, busy, need = float(p.get("low_pct", 10)), float(p.get("busy_pct", 50)), float(p.get("min_s", 60))
    period = float(p.get("sample_period_s", 10))
    latest = {k: s[-1] for k, s in h.gpu.items() if s}
    out = []
    for idx, ser in h.gpu.items():
        run = []
        for s in reversed(ser):
            if s[1] < low:
                run.append(s)
            else:
                break
        if not run:
            continue
        dur = run[0][0] - run[-1][0] + period
        others = [k for k, s in latest.items() if k != idx and s[1] >= busy]
        if dur >= need and others:
            out.append(Finding("gpu_util_drop", "warn",
                               f"GPU {idx} at {run[0][1]:.0f} % for {dur:.0f} s while GPU {others[0]} is busy",
                               [x[2] for x in reversed(run[:4]) if x[2]] or [f"GPU {idx} util {run[0][1]} %"],
                               p.get("action", "One rank may be hung or waiting; check the trainer log for that rank."),
                               key=f"gpu_util_drop:{idx}", escalate_to_judge=True))
    return out


# ---------------------------------------------------------------------------------------------- money
def credit_runway(t, v, h, cfg):
    p = P(cfg, "credit_runway")
    c, plan, dl = v.get("credit_usd"), v.get("planned_rate_usd_h"), v.get("deadlines") or {}
    if c is None:
        return None
    need = dl.get("train_stop")
    if not need or not plan:
        return []
    need += float(p.get("margin_min", 75)) * 60
    obs = None
    pts = [x for x in h.credit if t - x[0] <= float(p.get("window_h", 2)) * 3600]
    if pts:
        old = pts[0]
        if t - old[0] >= float(p.get("min_span_min", 25)) * 60 and old[1] > c:
            obs = (old[1] - c) / ((t - old[0]) / 3600)
    ends = {"planned": t + c / plan * 3600}
    if obs:
        ends["observed"] = t + c / obs * 3600
    which = min(ends, key=ends.get)
    if ends[which] >= need:
        return []
    rate = obs if which == "observed" else plan
    ev = [f"credit ${c:.2f} at {fmt12(t)}", f"planned rate ${plan:.2f}/h", f"observed burn {'$%.2f/h' % obs if obs else 'n/a'}",
          f"needed until {fmt12(need)} (training stop + {int(p.get('margin_min', 75))} min)"]
    return [Finding("credit_runway", "critical",
                    f"Provider credit ${c:.2f} runs out about {fmt12(ends[which])} at the {which} ${rate:.2f}/h, before {fmt12(need)}",
                    ev, tmpl(p.get("action", "Add credit (top up) now or plan an early stop; at $0 the provider stops the box.")))]


def ssh_box(t, v, h, cfg):
    """Login to the training box failing; plus the expensive combo: billing but unreachable."""
    p = P(cfg, "ssh_box")
    s = v.get("ssh")
    if s is None:
        return None
    if s.get("ok"):
        h.counters["ssh_fail"] = 0
        return []
    h.counters["ssh_fail"] = h.counters.get("ssh_fail", 0) + 1
    tail = [x for x in s.get("stderr_tail") or [] if x.strip()]
    last = tail[-1] if tail else f"rc {s.get('rc')}"
    auth = any("Permission denied" in x for x in tail)
    out = []
    if auth or h.counters["ssh_fail"] >= int(p.get("min_failures", 2)):
        out.append(Finding("ssh_box_down", "warn", f"Box ssh login fails: {last}", tail[-3:],
                           tmpl(p.get("action", "Check the ssh key attached to the instance and the box's sshd."))))
    for inst in v.get("instances") or []:
        if str(inst.get("actual_status")) != "running":
            continue
        up = t - float(inst.get("start_date") or t)
        if up < float(p.get("billing_grace_min", 10)) * 60:
            continue
        dph = float(inst.get("dph_total") or 0)
        ev = tail[-2:] + [f"instance {inst.get('id')} actual_status running, gpu_util {inst.get('gpu_util')}, "
                          f"up {mins(up)} min, ${dph:.2f}/h"]
        out.append(Finding("billing_but_unreachable", "critical",
                           f"Instance {inst.get('id')} bills ${dph:.2f}/h but the box cannot be reached ({mins(up)} min since start)",
                           ev, tmpl(p.get("billing_action", "Fix the ssh key on the instance, or destroy it; each idle hour costs ${dph}."),
                                    id=inst.get("id"), dph=f"{dph:.2f}"),
                           key=f"billing_but_unreachable:{inst.get('id')}"))
    return out


def done_but_billing(t, v, h, cfg):
    p = P(cfg, "done_but_billing")
    rc, inst = v.get("run_complete"), v.get("instances")
    if rc is None or inst is None:
        return None
    if not rc.get("complete"):
        h.forget("run_complete")
        return []
    since = h.seen("run_complete", t)
    out = []
    for i in inst:
        if str(i.get("actual_status")) not in ("running", "loading", "starting") or t - since < float(p.get("grace_min", 5)) * 60:
            continue
        dph = float(i.get("dph_total") or 0)
        out.append(Finding("done_but_still_billing", "critical",
                           f"Run is finished and saved, but instance {i.get('id')} is still running at ${dph:.2f}/h "
                           f"({mins(t - since)} min after completion was seen)",
                           [f"{rc.get('detail', 'run complete')} (seen {fmt12(since)})",
                            f"instance {i.get('id')}: actual_status {i.get('actual_status')}, gpu_util {i.get('gpu_util')}"],
                           tmpl(p.get("action", "Destroy it now: `vastai destroy instance {id}` (the watcher never destroys)."),
                                id=i.get("id")), key="done_but_still_billing"))
    return out


# ---------------------------------------------------------------------------------------------- transfers
def published_not_pulled(t, v, h, cfg):
    p = P(cfg, "published_not_pulled")
    ob = v.get("outbox")
    if ob is None:
        return None
    for it in v.get("pulled") or []:
        h.pulled.add(it)
    wait = [o for o in ob if o["item"] not in h.pulled and o.get("published_t")]
    if not wait:
        return []
    wait.sort(key=lambda o: o["published_t"])
    old = wait[0]
    age = t - old["published_t"]
    if age < float(p.get("max_wait_min", 60)) * 60:
        return []
    ev = [f"{o['item']} published {fmt12(o['published_t'])}" for o in wait[:6]]
    return [Finding("published_not_pulled", "critical",
                    f"Published save {old['item']} has waited {mins(age)} min without reaching the NAS ({len(wait)} waiting)",
                    ev, tmpl(p.get("action", "Check the puller (auth, container running); box disk fills while saves wait.")))]


def auth_errors(t, v, h, cfg):
    p = P(cfg, "auth_errors")
    a = v.get("auth_errors")
    if a is None:
        return None
    out = []
    for e in a:
        if int(e.get("count") or 0) > 0:
            out.append(Finding("puller_auth_errors", "warn",
                               f"{e['source']}: 'Permission denied (publickey)' x{e['count']} in 15 min",
                               e.get("sample", [])[-2:], tmpl(p.get("action", "Re-attach the key for {source}."), source=e["source"]),
                               key=f"puller_auth_errors:{e['source']}"))
    return out


def transfers(t, v, h, cfg):
    """A partial upload crawling (ETA far beyond plan) and finished files left behind it."""
    p = P(cfg, "transfers")
    tr = v.get("transfers")
    if tr is None:
        return None
    parts, srcs = tr.get("partials") or {}, tr.get("sources") or {}
    out, in_flight = [], set()
    for pp, size in parts.items():
        d, b = os.path.split(pp)
        src = os.path.join(d, b[1:-5] if b.startswith(".") and b.endswith(".part") else b)
        in_flight.add(src)
        first = h.seen(f"part:{pp}", t)
        ser = h.series.setdefault(f"part:{pp}", __import__("collections").deque(maxlen=50))
        if not ser or ser[-1][0] < t:
            ser.append([t, size])
        total = srcs.get(src)
        age = t - first
        if total is None or age < float(p.get("min_age_min", 60)) * 60 or len(ser) < 2:
            continue
        rate = (ser[-1][1] - ser[0][1]) / max(1.0, ser[-1][0] - ser[0][0])
        eta_h = (total - size) / rate / 3600 if rate > 0 else float("inf")
        if eta_h > float(p.get("max_eta_h", 2)):
            ev = [f"{pp}: {size} B (+{ser[-1][1] - ser[0][1]} B in {mins(ser[-1][0] - ser[0][0])} min)",
                  f"source {src}: {total} B; rate {rate / 1e6:.2f} MB/s"]
            out.append(Finding("artifact_upload_stalled", "warn",
                               f"Upload of {os.path.basename(src)} is crawling: {size / 1e9:.2f} of {total / 1e9:.2f} GB "
                               f"after {mins(age)} min, ETA {eta_h:.1f} h", ev,
                               tmpl(p.get("action", "Newer saves are queued behind it: copy the saves directly (skip optimizer "
                                          "state) or pause this upload.")), key=f"artifact_upload_stalled:{src}"))
    back = []
    for s, size in srcs.items():
        if size is None or s in in_flight or s in parts or s in (tr.get("dest") or {}):
            if size is None:
                h.forget(f"src:{s}")
            continue
        first = h.seen(f"src:{s}", t)
        if t - first >= float(p.get("backlog_min", 60)) * 60:
            back.append((first, s))
    if back:
        back.sort()
        f0, s0 = back[0]
        out.append(Finding("unsynced_saves_backlog", "critical",
                           f"{os.path.basename(s0)} has waited {mins(t - f0)} min on the host without reaching the NAS "
                           f"({len(back)} file(s) queued)", [f"{s} first seen {fmt12(f)}" for f, s in back[:6]],
                           tmpl(p.get("backlog_action", "Copy the waiting saves now; they are lost if the host goes away."))))
    return out


# ---------------------------------------------------------------------------------------------- generic sources
def process_gone(t, v, h, cfg):
    """A watched training process disappeared while the run is not marked done."""
    p = P(cfg, "process_gone")
    procs = v.get("processes")
    if procs is None:
        return None
    if v.get("_done") or (v.get("progress") or {}).get("done"):
        return []
    out = []
    for name, st in procs.items():
        c = h.counters.setdefault("proc_miss", {})
        if st.get("alive"):
            c[name] = 0
            h.counters.setdefault("proc_seen", {})[name] = True
            continue
        c[name] = c.get(name, 0) + 1
        if c[name] < int(p.get("min_cycles", 2)):
            continue
        seen = (h.counters.get("proc_seen") or {}).get(name)
        ev = recent_errors(h, t) or [f"{s}: {ln}" for s, tail in h.log_tail.items() for ln in list(tail)[-3:]][-6:]
        out.append(Finding("process_gone", "critical" if seen else "warn",
                           f"Training process '{name}' is not running" + ("" if seen else " (never seen since the watcher started)")
                           + f" and the run is not marked done ({c[name]} checks)", ev,
                           tmpl(p.get("action", "Read the last lines of the trainer log: crash, OOM kill or finished without a done marker? "
                                      "Resume from the newest checkpoint if it crashed.")), key=f"process_gone:{name}"))
    return out


def checkpoint_stale(t, v, h, cfg):
    """No new checkpoint for longer than the source's stale_min while training rows still arrive."""
    p = P(cfg, "checkpoint_stale")
    cks = v.get("checkpoints")
    if cks is None:
        return None
    out = []
    first = h.seen("checkpoint_watch", t)
    for name, c in cks.items():
        lim = float(c.get("stale_min") or p.get("stale_min", 90)) * 60
        now = float(c.get("now") or t)
        newest = c.get("newest_mtime")
        age = now - newest if newest else t - first
        if age <= lim or (v.get("progress") or {}).get("done"):
            continue
        if h.last_train_step is None:
            continue   # nothing trained yet: nothing to save
        out.append(Finding("checkpoint_stale", "warn",
                           f"No new checkpoint in '{name}' for {mins(age)} min (limit {int(lim // 60)} min) while training continues",
                           [f"newest: {c.get('newest') or 'none'} ({fmt12(newest) if newest else 'never'}), {c.get('count', 0)} total",
                            f"last update {h.last_train_step}"],
                           tmpl(p.get("action", "Check the save step (disk full? save crashed? save interval?). A crash now loses "
                                      "everything since the last save.")), key=f"checkpoint_stale:{name}", metric=age))
    return out


def gpus_idle(t, v, h, cfg):
    """Every GPU idle for a long stretch after training started: paying for nothing."""
    p = P(cfg, "gpus_idle")
    if not v.get("gpu_samples") or h.last_train_step is None:
        return None
    if (v.get("progress") or {}).get("done"):
        return []
    low, need = float(p.get("low_pct", 5)), float(p.get("idle_min", 20)) * 60
    spans = []
    for idx, ser in h.gpu.items():
        run = []
        for s in reversed(ser):
            if s[1] < low:
                run.append(s)
            else:
                break
        if not run or len(run) == len(ser) and len(ser) < 3:
            return []
        spans.append(run[0][0] - run[-1][0])
    if not spans or min(spans) < need:
        return []
    rate = v.get("planned_rate_usd_h")
    return [Finding("gpus_idle", "warn", f"All {len(spans)} GPUs below {low:.0f} % for {mins(min(spans))} min after training started"
                    + (f" (${rate:.2f}/h)" if rate else ""),
                    [f"GPU {k}: {s[-1][1]:.0f} %" for k, s in sorted(h.gpu.items()) if s][:8] + recent_errors(h, t, n=3),
                    tmpl(p.get("action", "Is the trainer hung (dataloader, NCCL) or finished? Check its log and process.")),
                    escalate_to_judge=True)]


def source_unreadable(t, v, h, cfg):
    """A configured source failed several cycles in a row (ssh refused, file gone, CLI broken). Unreadable is not healthy."""
    p = P(cfg, "source_unreadable")
    errs = v.get("source_errors")
    if errs is None:
        return None
    c = h.counters.setdefault("src_fail", {})
    for k in list(c):
        if k not in errs:
            c[k] = 0
    out = []
    for name, tail in errs.items():
        c[name] = c.get(name, 0) + 1
        if c[name] < int(p.get("min_cycles", 3)):
            continue
        tail = [x for x in tail or [] if str(x).strip()]
        auth = any("Permission denied" in str(x) or "publickey" in str(x) for x in tail)
        out.append(Finding("source_unreadable", "warn", f"Cannot read source '{name}' ({c[name]} checks in a row): "
                           + (str(tail[-1])[:160] if tail else "no detail"), [str(x)[:300] for x in tail[-3:]],
                           tmpl(p.get("auth_action" if auth else "action",
                                      "ssh key refused: check the key / host." if auth else
                                      "Check the path / host / CLI; the watcher is blind to this source until it reads again.")),
                           key=f"source_unreadable:{name}"))
    return out


def run_state(t, v, h, cfg):
    """Experiment tracker says the run crashed / failed (e.g. W&B run state)."""
    rs = v.get("run_state")
    if rs is None:
        return None
    out = []
    for name, st in rs.items():
        s = str(st.get("state") or "")
        if s in ("crashed", "failed", "killed"):
            out.append(Finding("run_state", "critical", f"Tracker reports run {st.get('run')} as {s.upper()} ({name})",
                               [f"{name}: state {s}"] + recent_errors(h, t, n=4),
                               "Read the trainer log tail; resume from the newest checkpoint if the box is still up.",
                               key=f"run_state:{name}"))
    return out


BUILTIN = [flags, log_patterns, loss_nonfinite, loss_spike, train_stalled, progress_reset, val_trend, disk_low,
           gpu_util_drop, credit_runway, ssh_box, done_but_billing, published_not_pulled, auth_errors, transfers,
           process_gone, checkpoint_stale, gpus_idle, source_unreadable, run_state]


def enabled(cfg: dict, extra: list | None = None) -> list:
    rules = [r for r in BUILTIN if P(cfg, r.__name__).get("enabled", True)]
    return rules + list(extra or [])
