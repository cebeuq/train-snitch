# Harness integrations

One MCP server (`train-snitch-mcp`) and one skill, packaged for each agent harness. Overview, install and the safety
model: [docs/HARNESSES.md](../docs/HARNESSES.md).

| Folder | Harness | What is in it |
|---|---|---|
| [claude-code](claude-code) | Claude Code | plugin (`.claude-plugin/`, `.mcp.json`, skill); marketplace at the repo root |
| [copilot](copilot) | GitHub Copilot CLI | `mcp-config.json`, skill |
| [cursor](cursor) | Cursor editor + `agent` CLI | `mcp.json`, skill, optional rule |
| [codex](codex) | OpenAI Codex CLI / IDE / app | `config.toml` snippet, skill, optional `agents/openai.yaml` |
| [antigravity](antigravity) | Google Antigravity (`agy`) | plugin (`plugin.json`, `mcp_config.json`, skill) |
| [gemini-cli](gemini-cli) | Gemini CLI | extension (`gemini-extension.json`, `GEMINI.md`, skill, `/snitch:status`) |
| [opencode](opencode) | OpenCode | `opencode.json` snippet, skill, `/snitch-status` |
| [skill](skill) | all | the canonical `SKILL.md` |

Edit only `skill/train-snitch/SKILL.md`, then run `python3 integrations/sync-skill.py`; a test checks every copy.
