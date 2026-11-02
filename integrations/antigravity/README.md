# train-snitch for Google Antigravity (`agy` CLI and app)

This folder is an Antigravity plugin: `plugin.json`, `mcp_config.json` (the MCP server) and `skills/`.

0. Get the repo files (the plugin/skill files below live in a clone, not in the pip package):
   `git clone https://github.com/cebeuq/train-snitch && cd train-snitch/integrations/antigravity`
1. Install the server once: `pipx install "train-snitch[mcp] @ git+https://github.com/cebeuq/train-snitch"`
2. Install the plugin from a clone of this repo:
   ```bash
   agy plugin validate . --strict      # path first: agy 1.1.5 reads a leading --strict as the path
   agy plugin install .
   agy plugin enable train-snitch      # asks for the config and output folders
   ```
   The plugin declares the two folders as `variables`, so it stays disabled until you answer them.
   `validate` fails until `train-snitch-mcp` is on `PATH` (step 1).

Without the plugin: add the `train-snitch` entry of [mcp_config.json](mcp_config.json) (with literal paths instead of
the `${...}` values) to `~/.gemini/config/mcp_config.json`, and copy `skills/train-snitch` to `~/.agents/skills/`.

Judge API keys: the plugin supports Pi's own login (`pi`, then `/login`) only. To use an API key file, add
`"TRAIN_SNITCH_ENV_FILE": "/ABS/PATH/train-snitch.env"` to the server's `env` by hand in
`~/.gemini/config/mcp_config.json` (the "without the plugin" route above). Never put keys in the plugin.
A Gemini CLI extension can also be imported: `agy plugin import gemini` (see ../gemini-cli).
