# train-snitch for Cursor (editor and `agent` CLI)

1. Install the server once: `pipx install "train-snitch[mcp] @ git+https://github.com/cebeuq/train-snitch"`
2. MCP: merge [mcp.json](mcp.json) into `~/.cursor/mcp.json` (all projects) or `.cursor/mcp.json` (one project).
   It reads `TRAIN_SNITCH_CONFIG_DIRS` / `TRAIN_SNITCH_OUT_DIRS` from your environment (`${env:...}`); you can also
   write the absolute paths in directly. The CLI uses the same file; approve it once with
   `agent mcp enable train-snitch`.
3. Skill: `mkdir -p ~/.cursor/skills && cp -r skills/train-snitch ~/.cursor/skills/`
   (or `.cursor/skills/` / `.agents/skills/` in a project).
4. Optional rule that nudges the agent to the tools: `cp rules/train-snitch.mdc .cursor/rules/`.

Use it: `/train-snitch`, or ask about your training run.
