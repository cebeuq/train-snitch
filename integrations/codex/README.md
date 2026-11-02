# train-snitch for OpenAI Codex (CLI, IDE extension, app)

1. Install the server once: `pipx install "train-snitch[mcp] @ git+https://github.com/cebeuq/train-snitch"`
2. Register it:
   ```bash
   codex mcp add train-snitch --env TRAIN_SNITCH_CONFIG_DIRS=/ABS/PATH/configs \
     --env TRAIN_SNITCH_OUT_DIRS=/ABS/PATH/runs -- train-snitch-mcp
   ```
   or paste [config.toml](config.toml) into `~/.codex/config.toml` (it forwards the variables from your shell with
   `env_vars`). A project-level `.codex/config.toml` works in trusted projects.
3. Add the skill (Codex reads `.agents/skills` in a repo and `~/.agents/skills` for the user):
   ```bash
   mkdir -p ~/.agents/skills && cp -r skills/train-snitch ~/.agents/skills/
   cp -r agents ~/.agents/skills/train-snitch/      # optional display metadata
   ```

Use it: `$train-snitch`, or just ask "any new train-snitch alerts?". Check with `codex mcp list`.

If the judge uses an API key and Codex does not pass it through, add its name to `env_vars`
(for example `"ANTHROPIC_API_KEY"`) or use `TRAIN_SNITCH_ENV_FILE` (docs/HARNESSES.md).
