# train-snitch

A read-only watchdog for neural-network training runs. Every minute, plain code reads the run (logs, metrics,
GPUs, disk, checkpoints, cloud billing). A small, cheap LLM (GPT-6 Luna by default, through the
[Pi](https://www.npmjs.com/package/@earendil-works/pi-coding-agent) agent harness) then reads a short digest of
what changed and decides: **ok**, **watch**, or **snitch**. Snitches and rule alerts go to a bigger agent
(Claude Code, Codex) or a human: an append-only alerts file, a desktop notification, or a webhook.

It never changes the run: no stop, no kill, no destroy. It only reads and reports. It exits by itself when the
run ends or at a deadline.

- **Any agent harness can drive it** through a small MCP server (start, stop, list, status, alerts): Claude Code,
  GitHub Copilot CLI, Cursor, OpenAI Codex, Google Antigravity, plus Gemini CLI and OpenCode.
  See [docs/HARNESSES.md](docs/HARNESSES.md).
- **Any LLM Pi can reach can be the judge**: Codex/Luna (default), Anthropic, Gemini, OpenAI, OpenRouter, Groq,
  Bedrock, Azure, Vertex, local Ollama / vLLM / llama.cpp / LM Studio, your own gateway, with fallbacks and spend
  caps. See [docs/JUDGE-MODELS.md](docs/JUDGE-MODELS.md).

```
every cycle (default 60 s)
  probes    read-only: one batched file read per host (local, or over ssh), plus fixed commands
            (nvidia-smi / rocm-smi, dmesg, ps, vastai / runpodctl, W&B API), each on its own interval
  adapter   probe output -> one normalized snapshot ("view")
  rules     free, deterministic: NaN/inf loss, loss spikes, val trend, stalls, crashes / tracebacks / NCCL
            timeouts / Xid / OOM, process gone, GPUs idle or one rank idle, disk, stale checkpoints,
            stuck uploads, credit runway, billing while unreachable or after the run is done, unreadable sources
  dedupe    new / ongoing / worsened / escalated / resolved; hysteresis; sparse reminders
  judge     small LLM via Pi, every `every_s` (60 s): digest + deltas -> ok / watch / snitch;
            a new rule alert makes that minute's call a Pi agent session with read-only tools
  guard     per-run Codex credit cap (default 150) + repeated-error back-off -> rules-only + one alert
  channels  alerts.jsonl (append-only) + latest.txt + status.json + macOS notification + webhook / command
```

## Quick start

Needs Python 3.10+ (stdlib only, nothing to install). The judge also needs Node and Pi
(`npm i -g @earendil-works/pi-coding-agent`, logged in to a provider; Luna uses Pi's Codex / ChatGPT login).
Without Pi, use `--rules-only`.

Install as a tool (optional; the `mcp` extra adds the MCP server for agent harnesses):

```bash
pipx install "train-snitch[mcp] @ git+https://github.com/cebeuq/train-snitch"   # train-snitch + train-snitch-mcp
```

Or run from a checkout:

```bash
git clone https://github.com/cebeuq/train-snitch && cd train-snitch
python3 -I -m unittest discover -s tests       # offline, no model calls (ffmpeg needed for the sheet tests)
# or everything (MCP SDK, Node + Pi bridge tests, ffmpeg) in a container:
# docker build -f tests/docker/Dockerfile -t train-snitch-test . && docker run --rm --network none train-snitch-test
cp examples/generic/remote-box.json my-run.json   # edit hosts / paths / sources
python3 -I bin/train-snitch run my-run.json --once --rules-only --dry     # one cycle: no LLM, no notifications
python3 -I bin/train-snitch run my-run.json                              # foreground, until the run ends
```

To start it detached (finite: it still exits on its own), use `setsid nohup ... &`. macOS has no `setsid`, so
use `perl -MPOSIX -e 'POSIX::setsid(); exec @ARGV' nohup python3 -I bin/train-snitch run my-run.json >> out.log 2>&1 &`.
To stop early: `kill "$(cat <out_dir>/watcher.pid)"`.

Other commands:

| Command | What |
|---|---|
| `train-snitch replay CONFIG INCIDENT_DIR` | replay a recorded incident through the rules and print the alerts |
| `train-snitch tail OUT_DIR [--min warn\|critical] [--all]` | follow the alerts (for an agent's Monitor tool) |
| `train-snitch meter` | Codex usage meter + credit balance, without a model call |
| `train-snitch check CONFIG` | validate a config and print the judge model chain (no model call) |
| `train-snitch run CONFIG --overrides JSON` | merge a small JSON object over the config (e.g. a judge model) |
| `train-snitch mcp` / `train-snitch-mcp` | MCP stdio server for agent harnesses (needs the `mcp` extra) |

## Quick start per agent harness

Install `train-snitch-mcp` (above), point it at your folders, register it, and add the skill. Full steps in each
folder's README under [integrations/](integrations):

```bash
export TRAIN_SNITCH_CONFIG_DIRS=/ABS/PATH/configs TRAIN_SNITCH_OUT_DIRS=/ABS/PATH/runs
claude plugin marketplace add cebeuq/train-snitch && claude plugin install train-snitch@train-snitch   # Claude Code
codex mcp add train-snitch --env TRAIN_SNITCH_CONFIG_DIRS=/ABS/PATH/configs -- train-snitch-mcp        # Codex
copilot mcp add train-snitch --env TRAIN_SNITCH_CONFIG_DIRS=/ABS/PATH/configs -- train-snitch-mcp      # Copilot CLI
git clone https://github.com/cebeuq/train-snitch && cd train-snitch    # Antigravity / Gemini CLI install from a clone
agy plugin install ./integrations/antigravity && agy plugin enable train-snitch                       # Antigravity
gemini extensions install ./integrations/gemini-cli                                                    # Gemini CLI
# Cursor: integrations/cursor/mcp.json -> ~/.cursor/mcp.json; OpenCode: integrations/opencode/opencode.json
```

Then ask the agent "start watching configs/my-run.json" or "any new train-snitch alerts?". Tools: `list_watches`,
`status`, `latest`, `tail_alerts`, `check_config`, `start_watch`, `stop_watch`, `meter`. Configs must sit under
`TRAIN_SNITCH_CONFIG_DIRS`, the server has no shell, and `stop_watch` only stops watchers, never the run.

## For the main agent: follow the alerts

In Claude Code, run this with the **Monitor** tool. Each printed line is one notification, and it exits when the
watcher exits:

```
python3 -I /path/to/train-snitch/bin/train-snitch tail /path/to/out_dir
```

(`out_dir` is the `out_dir` of your run config.)

It prints one line for each:

- new, escalated, or worsened warn+ alert
- critical reminder
- judge note saying a rule alert looks like a false alarm
- `WATCHER SILENT` line, if `status.json` stops updating for 15 min (hang)
- `WATCHER GONE` line, then exit, if the watcher process is gone without an exit record (killed -9)
- the one `TEST` alert, if the watcher was started with `--test-alert`

At the end it prints a `watcher exiting` line. `--min critical` narrows the output. `--all` adds ongoing
reminders.

Dependency-free alternative (tail until new line):

```
tail -n 0 -F /path/to/out_dir/alerts.jsonl | grep --line-buffered -E '"severity": "(warn|critical)"'
```

One-shot wait for Bash `run_in_background` ("tell me when anything critical happens"):

```
until grep -q '"severity": "critical"' /path/to/out_dir/alerts.jsonl; do sleep 30; done
```

Files in `out_dir`:

| File | What |
|---|---|
| `alerts.jsonl` | append-only, one JSON alert per line: the contract with the bigger agent |
| `latest.txt` | the newest warn+ alert in plain text, plus every still-active problem |
| `status.json` | heartbeat each cycle: active alerts, the judge's last verdict, the budget, the deadline |
| `judge.jsonl` | every judge call: verdict, tokens, latency, credits used (also ok / watch) |
| `state.json`, `watcher.log`, `watcher.pid` | internals; state survives a manual restart |

Alert record:

```json
{"time": "Nov 01 08:32 PM PDT", "severity": "critical", "status": "new", "rule": "loss_nonfinite",
 "what": "Training loss is NaN from update 6 (2 of 2 new rows)", "evidence": ["update 6: loss nan", "..."],
 "action": "Training has diverged and the box keeps billing. ...", "since": "Nov 01 08:32 PM PDT", "count": 1,
 "source": "rules", "key": "loss_nonfinite", "t": 1793503920}
```

- `severity`: `info`, `warn`, or `critical`.
- `status`:
  - `new`
  - `ongoing`: a reminder
  - `worsened`: the same metric at least 2x worse
  - `escalated`: the severity went up
  - `resolved`
- `source`:
  - `rules`
  - `judge`: `rule` is `judge:<topic>`, with `decision` and `assessment`. `about` lists the rule keys it judged
    a false alarm.
  - `watcher`: start, exit, `judge_off`, `watcher_cycle_error`, or the `test` record (`"test": true`).

## Robustness and secrets

- **A broken cycle does not kill the watcher.** An exception in the adapter, rules or judge becomes one deduped
  warn (`watcher_cycle_error`, with the traceback tail; `resolved` when a cycle works again). After
  `lifecycle.max_cycle_errors` (10) failed cycles in a row it exits. Every exit, crash included, writes a
  `watcher_exit` record (warn and pushed through every channel if it crashed), so `tail` always ends.
- **Secrets are masked** (`train_snitch/redact.py`) at ingest and again before every output: alerts, latest.txt,
  webhook, command hook, state.json, and everything sent to the LLM, including the output of the judge's
  read-only probe. Covered: `hf_`, `sk-`, GitHub (`ghp_`, `github_pat_`...), AWS `AKIA`, Slack `xox?-`, JWTs,
  `Bearer ...`, `key/token/password/secret=...` (also in URLs), `user:pass@` in URLs, PEM private keys, and 40-hex
  W&B keys next to "wandb".
- **Delivery check:** `run --test-alert` sends one info record labeled TEST through every configured channel
  (forced past `min_severity`).
- **Config reuse:** `{"extends": "config.json", ...}` merges a small per-run file over a base config.
  `lifecycle.deadline_local` (e.g. `"2026-11-01 08:00 AM"`) sets a wall-clock deadline.

## Configure a run (`adapter: "generic"`)

One JSON file is merged over the defaults in `train_snitch/config.py`. A run is a list of `sources`. Any source
can sit on a remote host (`"host": "box"`, declared under `hosts`, reached over ssh with `BatchMode`). All file
sources on one host are read by a single call of `train_snitch/reader.py`, sent over ssh's stdin. That call reads
only the bytes added since the last call, and it needs only `python3` on the remote host. See
`examples/generic/remote-box.json` and `examples/generic/local-run.json`.

| `type` | Reads | Notes |
|---|---|---|
| `log` | text log (path or glob, `all` for every match) | `format`: `auto`, `hf` (HF Trainer / TRL dict lines), `lightning` (progress bar), `deepspeed` (DeepSpeed / Megatron), `keyvalue` (`step 12 loss=0.5`: accelerate, torchrun, custom), `regex` (yours), `none`. Failure lines match the log patterns. |
| `jsonl` / `csv` | metric files | `step_key`, `loss_key`, `val_loss_key`, `kind` (train / val), `kind_key` |
| `tensorboard` | event files | `glob` or `path`; `loss_tag`, `val_tag`. Built-in TFRecord reader, no TensorFlow needed. |
| `wandb_dir` | local or offline W&B run dir | newest run's `wandb-summary.json` and `output.log` |
| `wandb_api` | W&B online | `run: entity/project/id`, `python` with the `wandb` package (your API key in its env) |
| `gpu` | `nvidia-smi` or `rocm-smi` (`tool`) | util, power, memory per GPU |
| `kernel_log` | `dmesg -T`, else `journalctl -k` | Xid, OOM killer, GPU fallen off the bus |
| `process` | `ps -eo pid,args` + `pattern` | `process_gone` fires after 2 misses when not done |
| `disk` | free space at `path` | |
| `checkpoints` | `glob` of saves | `stale_min`: no new save for that long while training runs |
| `transfer` | `source.dir` vs `dest.dir` (each may be on a host) | saves never reaching storage; partial uploads crawling |
| `vast` / `runpod` | `vastai show instances` / `runpodctl pod get` (+ credit / balance) | only allowlisted fields are kept |
| `done_file` | marker file | the watcher exits when it appears |
| `command` | any read-only argv | `parse: json` + `map` (`disk_free_gb`, `credit_usd`, `flags`, `done`), or plain lines as a log |

Other keys:

- `planned_rate_usd_h` and `train_stop_epoch` feed the credit-runway rule.
- `protected` lists container names that must never be stopped.
- `every_s` sets a per-type interval.

Top-level keys:

- `cycle_s`, `timezone` (all times are 12-hour AM/PM), `out_dir`
- `lifecycle.max_hours` / `deadline_epoch`
- `rules.<name>`: thresholds, or `{"enabled": false}`
- `log_patterns_extra`: your own regex alerts, with `severity`, `what`, `action`, `ignore`, `sources`, `min_count`, `judge`
- `flags`: status-flag files a run writes
- `engine`: `clear_cycles`, event TTL, reminder gaps
- `channels`: `notify` / `webhook` (`url` or `url_env`) / `command`, each with `min_severity`

For a setup the generic adapter can't describe, write an adapter class (`"adapter": "path/to/adapter.py:Class"`,
interface in `train_snitch/adapters/base.py`).

## Rules

Free and deterministic; each has thresholds under `rules.<name>` (or `{"enabled": false}`):
`flags` (status-flag files), `log_patterns` (crashes, tracebacks, NCCL timeouts, Xid, OOM, plus your own),
`loss_nonfinite`, `loss_spike`, `train_stalled`, `progress_reset`, `val_trend`, `disk_low`, `gpu_util_drop`,
`gpus_idle`, `credit_runway`, `done_but_billing`, `ssh_box`, `transfers`, `published_not_pulled`, `auth_errors`,
`process_gone`, `checkpoint_stale`, `source_unreadable`, `run_state`.

## The judge (Pi + Luna)

- **Built on Pi, no homemade agent loop.** `judge/pi_judge.mjs` makes one judgment per process.
  - Calm minutes: one stateless `ModelRuntime.complete()` call (Pi's model layer; no tools, no session, no
    history).
  - A minute with a new warn+ rule alert: an in-memory Pi `AgentSession` with no built-in tools, extensions,
    skills, or context files. It gets read-only tools over evidence the watcher already collected (`read_log`,
    `get_series`, `alert_history`). It may also get one adapter-declared read-only command (`run_probe`, at most
    once per judgment and once per 15 min). Turn cap: `max_turns` (default 3).
  - Codex models get strict JSON-schema output and `reasoning effort none` through Pi's request hook. The
    `x-codex-*` headers carry the credit balance.
  - Auth is Pi's own. Nothing here reads, copies, or prints credentials.
- **Prompt.** `judge/system-prompt.md` (short and fixed) plus a digest of about 0.5-3 KB:
  - run facts and active rule alerts
  - rows, losses, and validation since the last look
  - new error-looking log lines (benign noise filtered)
  - GPU, disk, credit, instances, processes, checkpoints, and unreadable sources
  - its own previous verdict, and the topics it already reported
- **Decisions.** `ok` and `watch` go only to `judge.jsonl` and `status.json`. A `snitch` becomes an alert, deduped
  by the model's `topic` slug:
  - new topic: `new`
  - same topic at a higher severity: `escalated`
  - the same topic again within 30 min: silent
  - after that: one `ongoing` reminder

  The judge is advisory and never suppresses a rule alert.
- **Cost guard.**
  - Cadence: `every_s` (default 60). Optionally `skip_unchanged`.
  - Codex credits are read at start (`codex app-server`, no model call), from each response header, and every 15
    judgments. When the balance falls more than `credit_cap` (default 150) during the run, the judge turns off for
    the rest of the run and writes one `judge_off` alert. Rules continue. The balance is account-wide, so other
    Codex use at the same time counts too (conservative).
  - 3 failed calls in a row, or a provider usage-limit refusal: same back-off.
  - `max_calls` (2000) is a hard backstop.
- **Measured.** Measured live: about 900-1,200 tokens in and about 120 out per call, 2-3 s. That is roughly 0.001
  Codex credits per call, about 1-2 credits for a 14 h run at one call a minute.
- **Other models.** `judge.model` = `provider/id` for any of Pi's providers, plus custom endpoints
  (`api`, `base_url`, `api_key_env` / `no_auth`, `compat`, `model_info`), `reasoning`, `structured_output`,
  `fallbacks` and `cost_cap_usd`. All of it maps onto Pi's own provider config; keys stay in Pi's login or env vars.
  Non-Codex models use Pi's portable `completeSimple()`; the default Codex request is unchanged. The Codex meter
  turns itself on only when an `openai-codex` model is in the chain. Details and 20 example configs:
  [docs/JUDGE-MODELS.md](docs/JUDGE-MODELS.md); background: `docs/PI-NOTES.md`.

## Eval judge (vision model on sample images)

`train_snitch/evaljudge.py` is a separate, generic module for any training that writes sample images or other image
eval artifacts. It is its own finite process (`train-snitch eval-judge CONFIG`), next to the watcher.

```
every poll_s (120 s)
  list    source folder, local or `ssh ... find` (e.g. a storage mirror); a regex on each relative path gives
          step + prompt (+ variant): ai-toolkit `<ms>__<step>_<idx>.jpg` (idx -> prompt, seed via group_size) or
          `<step>/pNNN.jpg`
  detect  a set is ready when it has `expect` images (or nothing new for `settle_s`)
  sheets  one row per prompt: REF (fixed reference, e.g. base renders) | PREV <step> S<k> | CUR <step> S<k>,
          4 rows per sheet; ffmpeg + built-in bitmap font (train_snitch/sheet.py, no Pillow)
  judge   ONE `codex exec -m gpt-6.1-terra --image <sheets> --output-schema` call per set (lean Codex setup)
          -> {step, structure_score 1-5, look_score 1-5,
          vs_previous{structure,look}, vs_reference{...}, previous_regressions_persist, regressions[], notable[],
          recommend, confidence, reason}. The previous verdict's regressions go into the prompt.
  guard   Codex credit balance read before/after every call (codex app-server, no model call); per-run cap
          `judge.credit_cap` (200); over it -> no more calls + one critical alert. 1 retry on a failed call.
  policy  decide(): STOP only on (a) structure "worse" vs previous, confirmed on the next set (also "worse", or
          with regression_confirm "persist": the judge says the regression persists and it is not "better"),
          or (b) no improvement in structure or look for 2 sets in a row at step >= plateau_min_step (1500).
          Verdicts below min_confidence (0.5) never trigger a stop. The model's own `recommend` is advice only.
  act     `act: "stop"`: on a STOP decision run `stop_command` ONCE (recorded in state.json before it runs; a
          restarted judge never runs it again). It never destroys anything. `--no-act` / `act: "none"`:
          recommendation only, keeps judging.
  report  every verdict -> alerts.jsonl (own out_dir + `alerts_jsonl`, e.g. the live watcher's file; severity
          info = holding/improving, warn = something got worse, critical = STOP decision), verdicts.jsonl,
          judge-log.md (human-readable, 12-hour times, sheets in sheets/<step>/), status.json, optional `event_log` command.
```

```bash
python3 -I bin/train-snitch eval-judge CONFIG --once --dry       # list + sheets, no model call
python3 -I bin/train-snitch eval-judge CONFIG --no-act           # judge, never run the stop command
python3 -I bin/train-snitch tail OUT_DIR                         # follow its alerts (Monitor tool)
```

`tail` on the watcher's out_dir also prints every eval-judge record (source `eval_judge`), whatever its severity.
The judge's exit record (`watcher_exit`) goes only to its own alerts file, so it never ends the watcher's tail.

Config example: `examples/eval-judge/generic-local.json`. A live call has one image sheet per 4 prompts, so expect
tens of thousands of tokens per set.

## Tests and the replay corpus

- **`tests/replay/`: 3 incidents.** All synthetic, in the frameworks' real log formats:
  - an HF Trainer under torchrun going NaN and crashing
  - a DeepSpeed rank hanging into an NCCL timeout
  - a healthy Lightning run with real noise (must stay quiet)

  Each incident checks the required alerts (rule, severity, status, text), the false alarms that must not
  happen, and the alert budget (dedupe). `tests/replay/tools/make_generic_incidents.py` regenerates them.

- **`tests/test_unit.py`:**
  - the engine state machine, rules, and channels
  - the judge: cadence, modes, topic dedupe, false-alarm notes, credit cap including after a restart, error
    back-off, usage limit
  - the real Pi bridge against Pi's offline `faux` provider
  - the probe runner, and an end-to-end watcher run with `tail`
- **`tests/test_fixes.py`:** fault injection (one adapter exception survived, repeated errors give up with an
  exit record, judge exception), `tail` exiting on a dead watcher, redaction everywhere, forced TEST delivery,
  and the webhook against a local server.
- **`tests/test_llm.py`:** judge model config validation, the mapping onto Pi per provider, the unchanged
  default request, fallback order / cooldown / spend cap, and the real bridge's model chain against Pi's faux
  provider and a fake OpenAI-compatible server (schema, auth header, 429 fallback).
- **`tests/test_harness.py`:** the control layer and its safety boundary, the MCP server through the official
  SDK (in-process and over stdio: start, status, tail, stop), and the integration files (identical skills,
  parseable configs, real tool names).
- **`tests/test_generic.py`:**
  - the log parsers
  - the file reader (tail offsets, CSV, TensorBoard simple and tensor scalars, W&B dir, stat, disk)
  - local plus remote sources through a fake `ssh` (fake `nvidia-smi`, `dmesg`, `vastai`, `runpodctl`)
  - an ssh `Permission denied` source turning into `source_unreadable`
  - a real process disappearing, giving `process_gone`
  - stale checkpoints and idle GPUs

## Design notes

- **Rules first, LLM second.** Most failures (NaN, crashes, flags, lockouts, money) need no model. The model is
  for connecting signals (e.g. GPUs busy but no rows; money burning, nothing moving) and for calling false alarms.
  One example: a smoke check that "failed" only on a benign NCCL warning.
- **Unreadable is not healthy.** A rule whose input could not be read returns None, and its alerts are held, not
  resolved. A source that fails 3 cycles in a row becomes its own alert, so a dead ssh never looks like "all
  clear".
- **Dedupe by key.**
  - Conditions resolve after `clear_cycles` clean cycles.
  - Events (log lines) stay active for 45 min, so a repeat is `ongoing`, not new.
  - Milestones fire once.
  - Reminders: critical every 30 min, warn every 2 h.
- **Read-only by construction.**
  - Probes are fixed argv lists (no shell, except the fixed `dmesg` / `journalctl` line).
  - The file reader opens files read-only.
  - The judge's only command is an allowlisted read-only probe whose argument must match a regex.
  - Only allowlisted provider fields are kept: no tokens, emails, or keys.
- **Finite.** One process, and one pid lock per output folder. It exits on run end, deadline, or SIGTERM.
  Nothing restarts it: no daemon, no LaunchAgent, no cron.

## Layout

```
train_snitch/   core: watcher, engine, rules, probes, reader, parsers, digest, judge, llm (judge model config),
                meter, channels, replay, control (harness control), mcp_server (MCP stdio server), CLI
  adapters/     base.py (interface), generic.py (declarative sources)
judge/          pi_judge.mjs (Pi bridge), system-prompt.md (shipped in the wheel as train_snitch/_judge)
integrations/   one MCP registration + skill per harness: claude-code, copilot, cursor, codex, antigravity,
                gemini-cli, opencode; skill/ (canonical SKILL.md), sync-skill.py
.claude-plugin/ marketplace.json (Claude Code plugin marketplace)
examples/       generic/, eval-judge/, judge-configs/ (20 judge model setups)
tests/          test_unit.py, test_fixes.py, test_generic.py, test_replay.py, test_llm.py, test_harness.py,
                replay/ (corpus + tools/), docker/Dockerfile (offline test image)
docs/           HARNESSES.md, JUDGE-MODELS.md, PI-NOTES.md (how the Pi integration was chosen)
bin/            train-snitch launcher
```
