# Replay corpus

Each folder is one incident the watcher should catch (or, for `generic-lightning-healthy`, must stay quiet on).
A replay test feeds `timeline.jsonl` to the watcher one cycle at a time and checks its alerts against `expected.json`.

## Files per incident

- `incident.json`: id, title, `source`, `adapter`, provenance, notes.
  - `source`: `synthetic` (built by the generator tool).
- `timeline.jsonl`: one line per watcher cycle: `{"t": epoch, "t_local": "...", "signals": {...}}`.
  Signal names are the adapter's normalized names (`adapter` in incident.json):
  - `now`: the cycle's epoch time.
  - `log_lines`: `{"<source id>": [new lines]}`.
  - `metric_rows`: `{"<source id>": [new metric rows]}`. Rows can contain bare `NaN`.
  - `processes`, `gpu`, and other per-source reads, as the generic adapter produces them.
  - `null` means "not readable this cycle". It is not the same as an empty value.
- `expected.json`:
  - `verdicts`: what the watcher must emit, each with `cycle` (index into the timeline), `severity`
    (`info`, `warn`, or `critical`), `rule` (a suggested rule id; `a|b` means either), and `status` (`new`,
    `ongoing`, `worsened`, `escalated`, or `resolved`).
  - `*_contains`: substrings that must appear in the alert's what / evidence / suggested action.
  - `escalate_to_judge`: true means the cheap rules should hand this to the small LLM judge.
  - `must_not`: alerts that would be false positives.
  - `max_alerts_warn_or_worse`: the dedupe budget for the whole timeline.
- Other files are the raw excerpts the timeline was built from.

## Sanitizing

The incidents here are synthetic, built with `tools/make_generic_incidents.py` (which regenerates them). Hosts use
documentation addresses (203.0.113.x) and paths are generic.
