"""Adapter interface.

An adapter knows one kind of run. It (1) declares read-only probes, (2) turns probe results into a *raw*
snapshot (the same shape the replay corpus stores in timeline.jsonl), and (3) normalizes a raw snapshot into
the *view* the built-in rules read. Live mode: probes -> collect -> normalize. Replay: timeline -> normalize.

View fields (all optional; None = not readable this cycle):
  flags            {NAME: value}               status flags written by the run
  state_line       str, state_line_new bool    the run's one-line state and whether it changed
  progress         {step, last_row_t, started_t, alive, done, now}
  train_rows       [{step, loss, raw}]         new training rows (loss may be NaN)
  val_rows         [{step, kind, loss, raw}]
  log_lines        {source: [new lines]}
  gpu_samples      [[t, idx, util_pct, power_w, raw_line]];  gpu_util_now {idx: pct}
  disk_free_gb     float
  credit_usd       float;  planned_rate_usd_h float;  deadlines {train_stop, cap_end} (epoch)
  instances        [{id, actual_status, dph_total, gpu_util, start_date, ...}]  ([] = gone)
  ssh              {ok, rc, stderr_tail}
  run_complete     {complete: bool, detail}
  outbox           [{item, published_t}];  pulled [item]
  auth_errors      [{source, count, sample}]
  transfers        {partials: {path: bytes}, sources: {path: bytes|None}, dest: {path: bytes}}
  protected        [container / process names that must not be stopped]
"""
from __future__ import annotations

from ..probes import Probe, Result


class Adapter:
    name = "base"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.acfg = cfg.get("adapter_config") or {}

    def probes(self) -> list[Probe]:
        return []

    def collect(self, results: dict[str, Result], hist, runner=None) -> dict:
        return {}

    def normalize(self, raw: dict, hist) -> dict:
        return {}

    def rules(self) -> list:
        return []

    def done(self, view: dict, hist, t: float) -> str | None:
        """Return a reason string when the run is over and the watcher should exit."""
        return None

    def deadline(self) -> float | None:
        """Optional epoch after which the watcher exits even if the run looks alive."""
        return None

    def judge_probes(self) -> list[dict]:
        """Extra read-only commands the LLM judge may call (at most one per judgment)."""
        return []

    def summary(self) -> dict:
        """Static run facts for the judge prompt and the status file (no secrets)."""
        return {}
