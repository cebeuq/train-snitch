# train-snitch for Google Antigravity (`agy` CLI and app)

This folder is an Antigravity plugin: `plugin.json`, `mcp_config.json` (the MCP server) and `skills/`.

1. Install the server once: `pipx install "train-snitch[mcp] @ git+https://github.com/cebeuq/train-snitch"`
2. Install the plugin from a clone of this repo:
   ```bash
   agy plugin validate --strict ./integrations/antigravity
   agy plugin install ./integrations/antigravity
   agy plugin enable train-snitch      # asks for the config and output folders
   ```
   The plugin declares the two folders as `variables`, so it stays disabled until you answer them.

Without the plugin: add the `train-snitch` entry of [mcp_config.json](mcp_config.json) (with literal paths instead of
the `${...}` values) to `~/.gemini/config/mcp_config.json`, and copy `skills/train-snitch` to `~/.agents/skills/`.

Judge API keys: use Pi's own login (`pi`, then `/login`) or `TRAIN_SNITCH_ENV_FILE`; do not put keys in the plugin.
A Gemini CLI extension can also be imported: `agy plugin import gemini` (see ../gemini-cli).
