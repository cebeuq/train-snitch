# train-snitch for Cursor (editor and `agent` CLI)

0. Get the repo files (the plugin/skill files below live in a clone, not in the pip package):
   `git clone https://github.com/cebeuq/train-snitch && cd train-snitch/integrations/cursor`
1. Install the server once: `pipx install "train-snitch[mcp] @ git+https://github.com/cebeuq/train-snitch"`
2. MCP: merge [mcp.json](mcp.json) into `~/.cursor/mcp.json` (all projects) or `.cursor/mcp.json` (one project).
   It reads `TRAIN_SNITCH_CONFIG_DIRS` / `TRAIN_SNITCH_OUT_DIRS` from your environment (`${env:...}`). Cursor's docs
   do not say what an unset `${env:...}` becomes, so writing the absolute paths in directly is the safer choice.
   For a judge API key file, add `"TRAIN_SNITCH_ENV_FILE": "/ABS/PATH/train-snitch.env"` (only if it exists). The CLI uses the same file; approve it once with
   `agent mcp enable train-snitch`.
3. Skill: `mkdir -p ~/.cursor/skills && cp -r skills/train-snitch ~/.cursor/skills/`
   (or `.cursor/skills/` / `.agents/skills/` in a project).
4. Optional rule that nudges the agent to the tools (run in your project, with the clone path filled in):
   `mkdir -p .cursor/rules && cp /PATH/TO/train-snitch/integrations/cursor/rules/train-snitch.mdc .cursor/rules/`.

Use it: `/train-snitch`, or ask about your training run.
