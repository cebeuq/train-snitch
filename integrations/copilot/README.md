# train-snitch for GitHub Copilot CLI

0. Get the repo files (the plugin/skill files below live in a clone, not in the pip package):
   `git clone https://github.com/cebeuq/train-snitch && cd train-snitch/integrations/copilot`
1. Install the server once: `pipx install "train-snitch[mcp] @ git+https://github.com/cebeuq/train-snitch"`
2. Register it:
   ```bash
   copilot mcp add train-snitch --env TRAIN_SNITCH_CONFIG_DIRS=/ABS/PATH/configs \
     --env TRAIN_SNITCH_OUT_DIRS=/ABS/PATH/runs -- train-snitch-mcp
   ```
   or merge [mcp-config.json](mcp-config.json) into `~/.copilot/mcp-config.json` and replace the `/ABS/PATH`
   placeholders. A project can commit the same block as `.github/mcp.json` (loaded after you trust the folder).
3. Skill: `mkdir -p ~/.copilot/skills && cp -r skills/train-snitch ~/.copilot/skills/`
   (or `.github/skills/` / `.agents/skills/` in a repo).

Copilot CLI passes MCP servers only `PATH` plus what `env` lists. The default judge uses Pi's own login (`pi`, then
`/login`), which needs no variables at all. If the judge needs an API key, either list it in `env`, or create a
private `chmod 600` file of `NAME=value` lines and add one more entry to `env` (only if the file exists; a missing
file makes every `start_watch` fail):

```json
"TRAIN_SNITCH_ENV_FILE": "/ABS/PATH/train-snitch.env"
```
