# train-snitch for Gemini CLI

This folder is a Gemini CLI extension: `gemini-extension.json` (the MCP server and three settings), `GEMINI.md`
(context), `skills/` and the `/snitch:status` command.

1. Install the server once: `pipx install "train-snitch[mcp] @ git+https://github.com/cebeuq/train-snitch"`
2. From a clone of this repo:
   ```bash
   gemini extensions install ./integrations/gemini-cli     # or: gemini extensions link ./integrations/gemini-cli
   ```
   It asks for the settings (config folders, output folders, optional env file). Local stdio servers need a
   trusted folder (`gemini trust`), or they show as disconnected.

Without the extension:

```bash
gemini mcp add -s user -e TRAIN_SNITCH_CONFIG_DIRS=/ABS/PATH/configs -e TRAIN_SNITCH_OUT_DIRS=/ABS/PATH/runs \
  train-snitch train-snitch-mcp
```

Gemini CLI hides sensitive-looking variables from MCP servers. For judge API keys use Pi's own login or
`TRAIN_SNITCH_ENV_FILE`. Antigravity users can import this extension with `agy plugin import gemini`.
