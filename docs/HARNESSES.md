# Driving train-snitch from an agent harness

Any coding agent that speaks MCP can start, stop, list and read train-snitch watchers through one small stdio
server, `train-snitch-mcp`. Each harness also gets the same `train-snitch` skill (a `SKILL.md` telling the agent
when and how to use the tools). The judge inside each watcher is unchanged: it still runs through Pi.

```
harness (Claude Code, Codex, ...) --MCP stdio--> train-snitch-mcp --spawns / reads--> watcher(s) --Pi--> judge LLM
                                                                                 out_dir/alerts.jsonl, status.json
```

## Which harnesses

Picked by work-usage share in the JetBrains Developer Ecosystem Survey 2026 (early results:
[JetBrains Research](https://blog.jetbrains.com/research/2026/08/ai-coding-agent-adoption-2026/), summarised by
[I Programmer](https://www.i-programmer.info/news/99-professional/19118-jetbrains-reveals-widespread-adoption-of-ai-coding-agents-.html):
Claude Code 39%, GitHub Copilot 21%, OpenAI Codex 16%, Cursor 12%). The fifth slot has no published share; it goes
to Google's current agent, Antigravity, because Gemini CLI was
["replaced by Antigravity CLI for unpaid-tier and Google One users on June 18th, 2026"](https://geminicli.com/docs/tools/mcp-server/).
Gemini CLI (still used on paid / enterprise plans) and OpenCode (popular open-source CLI, smaller share than the
four above) ship as extras, so whichever of the three is your fifth is covered:

| # | Harness | Package | Notes |
|---|---|---|---|
| 1 | Claude Code | [integrations/claude-code](../integrations/claude-code) | plugin + marketplace (`claude plugin install`) |
| 2 | GitHub Copilot (CLI) | [integrations/copilot](../integrations/copilot) | `copilot mcp add`, skill |
| 3 | Cursor (editor + `agent` CLI) | [integrations/cursor](../integrations/cursor) | `mcp.json`, skill, rule |
| 4 | OpenAI Codex (CLI / IDE / app) | [integrations/codex](../integrations/codex) | `codex mcp add` or `config.toml`, skill |
| 5 | Google Antigravity (`agy`) | [integrations/antigravity](../integrations/antigravity) | plugin; replaced Gemini CLI for unpaid and Google One users in June 2026 |
| + | Gemini CLI | [integrations/gemini-cli](../integrations/gemini-cli) | extension; still used on paid / enterprise plans, and `agy plugin import gemini` reads it |
| + | OpenCode | [integrations/opencode](../integrations/opencode) | `opencode.json`, skill, command |

All of them read the same `SKILL.md` format, so there is one canonical skill
(`integrations/skill/train-snitch/SKILL.md`) and real-file copies in each package (`integrations/sync-skill.py`).

## 1. Install the MCP server

The core is stdlib-only; the server needs the official MCP Python SDK (`mcp` 2.x), installed as an extra.
Use an isolated tool install so it never touches a system Python:

```bash
pipx install "train-snitch[mcp] @ git+https://github.com/cebeuq/train-snitch"
git clone https://github.com/cebeuq/train-snitch     # the plugin / skill / extension files live in the repo
# or without installing: uvx --from "train-snitch[mcp] @ git+https://github.com/cebeuq/train-snitch" train-snitch-mcp
```

That gives `train-snitch` (the CLI) and `train-snitch-mcp` (the server) on `PATH`. From a checkout, with the SDK
importable, `python3 -I bin/train-snitch mcp` runs the same server.

The judge still needs Node and Pi (`npm i -g @earendil-works/pi-coding-agent`) and a Pi login or API key; see
[JUDGE-MODELS.md](JUDGE-MODELS.md). Without them, start watchers with `rules_only`.

## 2. Tell it which folders it may use

| Variable | Meaning | Default |
|---|---|---|
| `TRAIN_SNITCH_CONFIG_DIRS` | folders (`:`-separated) whose `.json` run configs `start_watch` may use | the server's working directory |
| `TRAIN_SNITCH_OUT_DIRS` | folders that hold watcher `out_dir`s (searched 3 levels deep by `list_watches`) | the config folders |
| `TRAIN_SNITCH_ENV_FILE` | optional private (`chmod 600`) file of `NAME=value` lines added to each watcher's environment | none |

When the working directory is your home directory or `/`, nothing is allowed until the variables are set.
A config's `out_dir` must sit under an output folder (the default `"./out"` sits next to the config).

## 3. Register it in your harness

Each folder under [integrations/](../integrations) has a README with the exact steps. One-liners:

```bash
claude mcp add --scope user -e TRAIN_SNITCH_CONFIG_DIRS=/ABS/PATH/configs --transport stdio train-snitch -- train-snitch-mcp
codex mcp add train-snitch --env TRAIN_SNITCH_CONFIG_DIRS=/ABS/PATH/configs -- train-snitch-mcp
copilot mcp add train-snitch --env TRAIN_SNITCH_CONFIG_DIRS=/ABS/PATH/configs -- train-snitch-mcp
gemini mcp add -s user -e TRAIN_SNITCH_CONFIG_DIRS=/ABS/PATH/configs train-snitch train-snitch-mcp
```

Claude Code's `-e` takes several values, so keep another option (here `--transport stdio`) between the last `-e`
and the server name. Cursor (`~/.cursor/mcp.json`), OpenCode (`opencode.json`) and Antigravity (`mcp_config.json`
or the plugin) take the JSON files in their folders.

## Tools

| Tool | Does | Changes anything? |
|---|---|---|
| `list_watches()` | every watcher out_dir under the output folders: run name, alive, pid, last update, deadline, judge mode, active alerts, worst severity | no |
| `status(run)` | `status.json` plus liveness | no |
| `latest(run)` | `latest.txt` (newest warn+ alert and all active problems) | no |
| `tail_alerts(run, n, min_severity, since, include_ongoing)` | newest alert records, filtered like `train-snitch tail`. With `since=<next_since>`: the oldest `n` unread records and `more` (call again when true), so following never skips one | no |
| `check_config(config, overrides)` | validates a config and shows the judge model chain | no |
| `start_watch(config, overrides, rules_only, dry, once, test_alert)` | starts a detached, finite watcher (`train-snitch run`); returns within about 10 s, also with `once` | starts one process |
| `stop_watch(run)` | SIGTERM to that watcher; it writes its exit record | stops one watcher |
| `meter()` | Codex usage meter and credits (no model call) | no |

`run` is an out_dir path or a run name / folder name from `list_watches`.

## Safety model

train-snitch is read-only towards the training run, and the MCP server keeps it that way:

- **No shell.** No tool takes a command. The server only runs `train-snitch run` (and the Codex meter).
- **Configs come from the config folders.** A config can declare probe commands, a command hook and a custom
  adapter (Python code), so `start_watch` only accepts `.json` files under `TRAIN_SNITCH_CONFIG_DIRS`, and also
  checks every `extends` parent and a custom adapter's file against those folders. The agent using these tools
  usually has file-write tools too, so **keep `TRAIN_SNITCH_CONFIG_DIRS` outside any folder the agent can write
  to** (not inside the project or workspace it edits); otherwise it could write a config and start it.
- **Inline overrides are bounded.** `overrides` are limited to `name`, `cycle_s` (>= 5 s), `timezone`,
  `lifecycle` deadline keys, `rules` thresholds, and judge model / fallback strings / limits. Spend and lifetime
  caps (`credit_cap`, `cost_cap_usd`, `max_calls`, `max_turns`, `max_hours`) may only be lowered, `every_s` only
  raised, and the judge only switched off. Sources, hosts, channels, adapters, endpoints, binaries and keys are
  refused inline.
- **One watcher per out_dir.** The watcher holds an exclusive lock on `<out_dir>/watcher.lock` for its whole life,
  so two `start_watch` calls at the same moment (one or several MCP servers) cannot both run.
- **Only watchers are stopped.** `stop_watch` reads `<out_dir>/watcher.pid` and signals it only when that process's
  command line is a train-snitch watcher and it holds that out_dir's lock (so a pid reused after a crash is never
  signalled). It never sends SIGKILL and never touches the training job.
- **Folder allowlists.** Every path is resolved (symlinks too) and must sit under the allowed folders.
- **No secrets through the tools.** Alerts and status files are redacted by the watcher. API keys never appear in
  tool arguments or configs: the judge uses Pi's login or environment variables, and `TRAIN_SNITCH_ENV_FILE` values
  are passed to the watcher process only, never returned.
- **Finite.** Watchers exit at run end or their deadline. Nothing is made persistent (no daemon, no restart).
  A watcher started by the server outlives the session; stop it with `stop_watch` if you no longer need it.

## API keys and harness environments

Harnesses differ in what environment they give a stdio server. Copilot CLI passes only `PATH` plus listed
variables; Gemini CLI hides sensitive-looking ones; Codex forwards only `env_vars`; Antigravity documents no
pass-through. So, in order of preference:

1. **Pi's own login** (`pi`, then `/login`: ChatGPT/Codex, Anthropic, GitHub Copilot, ...). Stored by Pi in
   `~/.pi/agent/auth.json`; nothing to pass. This is how the default (GPT-6 Luna) works.
2. **`TRAIN_SNITCH_ENV_FILE`**: one private (`chmod 600`) file, e.g. `ANTHROPIC_API_KEY=...`, named in the server's
   `env`. Works in every harness whose MCP entry you can edit. The shipped Antigravity plugin and the Copilot / Cursor
   templates leave it out (a set-but-missing file makes `start_watch` fail); add it by hand when you have the file.
3. **The harness's env forwarding** (`env_vars` in Codex, `${VAR}` in Claude Code, `${env:VAR}` in Cursor,
   `{env:VAR}` in OpenCode).

If `HOME` is missing from the server's environment, the server fills it in so Pi can find its login.

## Without MCP

Every harness with a shell can use the CLI directly; the skill says so too:

```bash
train-snitch run CONFIG --once --rules-only --dry   # one test cycle
train-snitch tail OUT_DIR                           # one line per new alert; exits with the watcher
train-snitch check CONFIG                           # validate + show the judge model chain
```

## Verified and not verified

Tested offline in Docker: every tool through the SDK's in-process client and over real stdio
(start, status, tail, stop), the boundary checks, the env file, and that every integration file parses, registers
`train-snitch-mcp` and names only real tools; concurrent `start_watch` calls leave one watcher; following with
`since` loses no record. Config formats follow each vendor's current docs; the Claude Code plugin and marketplace
pass `claude plugin validate`. Not tested live in
each harness: whether Antigravity installs a plugin from a repo subfolder URL (install from a local path), and
whether Copilot expands `${VAR}` in `env` (the template uses literal paths).
