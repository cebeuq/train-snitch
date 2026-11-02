# train-snitch for OpenCode

1. Install the server once: `pipx install "train-snitch[mcp] @ git+https://github.com/cebeuq/train-snitch"`
2. MCP: merge the `mcp` block of [opencode.json](opencode.json) into `~/.config/opencode/opencode.json` (global) or a
   project's `opencode.json`. It reads the folders from your environment (`{env:...}`).
3. Skill and command:
   ```bash
   mkdir -p ~/.config/opencode/skills ~/.config/opencode/commands
   cp -r skills/train-snitch ~/.config/opencode/skills/
   cp commands/snitch-status.md ~/.config/opencode/commands/
   ```

Use it: `/snitch-status`, or ask about your training run.
