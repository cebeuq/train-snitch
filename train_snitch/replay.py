"""Replay a recorded incident (tests/replay/<id>/timeline.jsonl) through adapter.normalize + the rule engine.

No probes, no judge, no channels: deterministic and free. Used by the tests and by `train-snitch replay`.
"""
from __future__ import annotations

import json
import os

from . import config as cfgmod
from . import rules as rulemod
from . import timeutil
from .engine import Engine


def load_timeline(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(ln) for ln in f if ln.strip()]   # python json accepts bare NaN


def replay(incident_dir: str, cfg: dict) -> list[dict]:
    """Returns one entry per cycle: {t, t_local, records, active}."""
    timeutil.set_tz(cfg.get("timezone"))
    rulemod.set_vars(cfg.get("vars"))
    adapter = cfgmod.load_adapter(cfg)
    eng = Engine(rulemod.enabled(cfg, adapter.rules()), cfg)
    out = []
    for row in load_timeline(os.path.join(incident_dir, "timeline.jsonl")):
        view = adapter.normalize(row.get("signals") or {}, eng.hist)
        recs, active = eng.process(float(row["t"]), view)
        out.append({"t": row["t"], "t_local": row.get("t_local"), "records": recs, "active": active})
    return out
