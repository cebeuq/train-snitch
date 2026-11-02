---
name: train-snitch
description: Start, stop, list and check train-snitch watchers and read their alerts. train-snitch is a read-only watchdog for ML training runs (logs, GPUs, processes, cloud billing, plus an LLM judge). Use it when the user asks to watch a training run, whether a run is healthy, or for train-snitch alerts. Do not use it to stop, kill or change the training job itself.
---
# train-snitch

train-snitch watches a training run and never changes it. Every minute it reads logs, metrics, GPUs, disk,
checkpoints and cloud billing, runs fixed rules, and asks a small LLM judge (through Pi) for ok / watch / snitch.
Alerts go to `alerts.jsonl`, `latest.txt` and `status.json` in the run's `out_dir`.

## Tools (MCP server `train-snitch`)

- `list_watches()`: every watcher under the allowed folders: run name, out_dir, alive, last update, active alerts.
- `status(run)`: the heartbeat: active alerts, the judge's last verdict, budget, deadline.
  `run` is an out_dir path or a run name from `list_watches`.
- `latest(run)`: the newest warn+ alert in plain text plus every still-active problem.
- `tail_alerts(run, n=20, min_severity="warn", since=None)`: the newest alert records. To follow a run, call it
  again with `since` set to the `next_since` from the previous call; if the reply says `more: true`, call again
  at once (it returns the oldest unread records first, so none are skipped).
- `check_config(config)`: validate a run config and show the judge model chain. Runs nothing.
- `start_watch(config, rules_only=False, dry=False, once=False, test_alert=False, overrides=None)`: start a
  detached watcher for a config file. It is finite: it exits when the run ends or at its deadline.
  For a first try use `once=True, dry=True, rules_only=True`. Use `rules_only=True` when no LLM login is set up.
  It returns within about 10 s; read the results with `tail_alerts`. Inline `overrides` may lower caps and
  deadlines or switch the judge off, never raise them.
- `stop_watch(run)`: stop only that watcher. It never touches the training run.
- `meter()`: Codex usage and credits (no model call). Only matters for openai-codex judge models.

## Rules

1. Report every `critical` alert to the user at once. Quote its `what`, `evidence` and `action`.
2. Never stop, kill or destroy a training run, container or cloud instance because of an alert. Recommend it and
   let the user decide.
3. A judge record with `about` says a rule alert looks like a false alarm. Mention it next to that alert.
4. `WATCHER GONE` or a `watcher_exit` record with severity warn means the watcher crashed: tell the user, and offer
   `start_watch` again if the run is still live.
5. The judge model lives in the config's `judge` block (any Pi provider, local servers, fallbacks). Never put API
   keys in chat, configs or tool arguments; keys come from Pi's login or environment variables.
6. If the MCP tools are missing, use the CLI: `train-snitch tail OUT_DIR` to follow alerts, or
   `train-snitch run CONFIG --once --rules-only --dry` for one test cycle.
