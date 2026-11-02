# Driving train-snitch from an agent harness

Any coding agent that speaks MCP can start, stop, list and read train-snitch watchers through one small stdio
server, `train-snitch-mcp`. Each harness also gets the same `train-snitch` skill (a `SKILL.md` telling the agent
when and how to use the tools). The judge inside each watcher is unchanged: it still runs through Pi.

```
harness (Claude Code, Codex, ...) --MCP stdio--> train-snitch-mcp --spawns / reads--> watcher(s) --Pi--> judge LLM
                                                                                 out_dir/alerts.jsonl, status.json
```

## Which harnesses

Picked by share of developers' primary coding assistant (developer survey, Oct 2026), plus two cheap extras:

| # | Harness | Package | Notes |
|---|---|---|---|
| 1 | Claude Code | [integrations/claude-code](../integrations/claude-code) | plugin + marketplace (`claude plugin install`) |
| 2 | GitHub Copilot (CLI) | [integrations/copilot](../integrations/copilot) | `copilot mcp add`, skill |
| 3 | Cursor (editor + `agent` CLI) | [integrations/cursor](../integrations/cursor) | `mcp.json`, skill, rule |
| 4 | OpenAI Codex (CLI / IDE / app) | [integrations/codex](../integrations/codex) | `codex mcp add` or `config.toml`, skill |
| 5 | Google Antigravity (`agy`) | [integrations/antigravity](../integrations/antigravity) | plugin; replaced Gemini CLI for individual users in June 2026 |
| + | Gemini CLI | [integrations/gemini-cli](../integrations/gemini-cli) | extension; still used on paid / enterprise plans, and `agy plugin import gemini` reads it |
| + | OpenCode | [integrations/opencode](../integrations/opencode) | `opencode.json`, skill, command |

All of them read the same `SKILL.md` format, so there is one canonical skill
(`integrations/skill/train-snitch/SKILL.md`) and real-file copies in each package (`integrations/sync-skill.py`).

## 1. Install the MCP server

The core is stdlib-only; the server needs the official MCP Python SDK (`mcp` 2.x), installed as an extra.
Use an isolated tool install so it never touches a system Python:

```bash
pipx install "train-snitch[mcp] @ git+https://github.com/cebeuq/train-snitch"
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
claude mcp add --scope user -e TRAIN_SNITCH_CONFIG_DIRS=/ABS/PATH/configs train-snitch -- train-snitch-mcp
codex mcp add train-snitch --env TRAIN_SNITCH_CONFIG_DIRS=/ABS/PATH/configs -- train-snitch-mcp
copilot mcp add train-snitch --env TRAIN_SNITCH_CONFIG_DIRS=/ABS/PATH/configs -- train-snitch-mcp
gemini mcp add -s user -e TRAIN_SNITCH_CONFIG_DIRS=/ABS/PATH/configs train-snitch train-snitch-mcp
```

Cursor (`~/.cursor/mcp.json`), OpenCode (`opencode.json`) and Antigravity (`mcp_config.json` or the plugin) take the
JSON files in their folders.

## Tools

| Tool | Does | Changes anything? |
|---|---|---|
| `list_watches()` | every watcher out_dir under the output folders: run name, alive, pid, last update, deadline, judge mode, active alerts, worst severity | no |
| `status(run)` | `status.json` plus liveness | no |
| `latest(run)` | `latest.txt` (newest warn+ alert and all active problems) | no |
| `tail_alerts(run, n, min_severity, since, include_ongoing)` | newest alert records, filtered like `train-snitch tail`; returns `next_since` to follow | no |
| `check_config(config, overrides)` | validates a config and shows the judge model chain | no |
| `start_watch(config, overrides, rules_only, dry, once, test_alert)` | starts a detached, finite watcher (`train-snitch run`) | starts one process |
| `stop_watch(run)` | SIGTERM to that watcher; it writes its exit record | stops one watcher |
| `meter()` | Codex usage meter and credits (no model call) | no |

`run` is an out_dir path or a run name / folder name from `list_watches`.

## Safety model

train-snitch is read-only towards the training run, and the MCP server keeps it that way:

- **No shell.** No tool takes a command. The server only runs `train-snitch run` (and the Codex meter).
- **Configs come from files you placed.** A config can declare probe commands and a command hook, so
  `start_watch` only accepts `.json` files under `TRAIN_SNITCH_CONFIG_DIRS`. Inline `overrides` are limited to
  `name`, `cycle_s`, `timezone`, `lifecycle` deadline keys, `rules` thresholds, and judge model / fallback
  strings / limits. Sources, hosts, channels, adapters, endpoints, binaries and keys are refused inline.
- **Only watchers are stopped.** `stop_watch` reads `<out_dir>/watcher.pid` and signals it only when that process's
  command line is a train-snitch watcher. It never sends SIGKILL and never touches the training job.
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
2. **`TRAIN_SNITCH_ENV_FILE`**: one private file, e.g. `ANTHROPIC_API_KEY=...`; works in every harness.
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
`train-snitch-mcp` and names only real tools. Config formats follow each vendor's current docs. Not tested live in
each harness: whether Antigravity installs a plugin from a repo subfolder URL (install from a local path), and
whether Copilot expands `${VAR}` in `env` (the template uses literal paths).
