# train-snitch for Claude Code

A Claude Code plugin: the `train-snitch` MCP server plus the `train-snitch` skill.

1. Install the server once (see [docs/HARNESSES.md](../../docs/HARNESSES.md#1-install-the-mcp-server)):
   `pipx install "train-snitch[mcp] @ git+https://github.com/cebeuq/train-snitch"`
2. Tell it which folders it may use (in the shell you start `claude` from, or your shell profile):
   ```bash
   export TRAIN_SNITCH_CONFIG_DIRS=/ABS/PATH/configs   # run configs it may start
   export TRAIN_SNITCH_OUT_DIRS=/ABS/PATH/runs         # where watcher out_dirs live
   ```
3. Install the plugin from this repo's marketplace:
   ```
   claude plugin marketplace add cebeuq/train-snitch
   claude plugin install train-snitch@train-snitch
   ```
   Or from a clone: `claude plugin marketplace add ./train-snitch`.

Without the plugin (MCP server only):

```bash
claude mcp add --scope user --transport stdio \
  -e TRAIN_SNITCH_CONFIG_DIRS=/ABS/PATH/configs -e TRAIN_SNITCH_OUT_DIRS=/ABS/PATH/runs \
  train-snitch -- train-snitch-mcp
mkdir -p ~/.claude/skills && cp -r skills/train-snitch ~/.claude/skills/
```

Use it: "is my training run healthy?", "start watching configs/run.json", or `/train-snitch:train-snitch`.
Tools appear as `mcp__plugin_train-snitch_train-snitch__<tool>` (plugin) or `mcp__train-snitch__<tool>`.

Following a run live: Claude Code's Monitor tool can also run `train-snitch tail OUT_DIR` (one line per alert).
