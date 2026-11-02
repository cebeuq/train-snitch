# Judge models: any LLM Pi can reach

The judge always runs through Pi (`judge/pi_judge.mjs`). What changes is which model Pi calls. train-snitch has no
API clients of its own: every option below maps onto Pi's own provider config, the same fields as a provider entry
in Pi's `models.json`, registered in memory for the judge only (nothing is written to disk).

With none of the new keys set, the default (`openai-codex/gpt-6-luna`) sends Pi exactly the request it sent before.

## The `judge` block

| Key | Meaning | Pi equivalent |
|---|---|---|
| `model` | `"provider/id"`, the same string as `pi --model` (`pi --list-models` shows them) | `getModel(provider, id)` |
| `api` | custom endpoint only: `openai-completions`, `openai-responses`, `anthropic-messages`, `google-generative-ai` | provider `api` |
| `base_url` | endpoint URL. On a built-in provider: a proxy, catalog kept | provider `baseUrl` |
| `api_key_env` | the **name** of the env var holding the key | provider `apiKey: "$NAME"` |
| `no_auth` | keyless local server | `apiKey: "local"` (dummy, as Pi's docs advise) |
| `headers` | extra headers; secret values must be `"$ENV_NAME"` | provider `headers` |
| `compat` | Pi compatibility flags, passed unchanged | provider `compat` |
| `model_info` | custom models: `context_window`, `max_tokens`, `reasoning`, `input`, `cost: {input, output}` (USD per 1M tokens) | `models[0]` |
| `reasoning` | thinking level: `off`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max` | `reasoning` / `thinkingLevel` |
| `temperature` | left out when unset (reasoning models often reject it) | `temperature` |
| `max_tokens` | reply cap (default 500) | `maxTokens` |
| `request_timeout_s` | per-request timeout inside Pi (default 60) | `timeoutMs` |
| `structured_output` | `auto`, `schema` or `prompt` (below) | request hook |
| `fallbacks` | ordered list of `"provider/id"` strings or objects with the keys above | more candidates |
| `fallback_on` | failure kinds that move to the next candidate (default `not_found`, `no_auth`, `error`, `rate_limit`, `timeout`; also `unparseable`, `config`) | |
| `fallback_cooldown_min` | after `rate_limit` / `no_auth` / `not_found`, skip that candidate this long (default 15) | |
| `cost_cap_usd` | per-run spend cap from Pi's own per-call cost figure; above it the judge turns off (rules continue) | `usage.cost.total` |
| `meter` | `auto` (default: Codex meter when an `openai-codex` model is in the chain), `codex`, `none` | |
| `pi_agent_dir` | Pi config folder (login, models.json) for the judge | `PI_CODING_AGENT_DIR` |

Unchanged: `enabled`, `every_s`, `credit_cap` (Codex credits), `max_calls`, `max_turns`, `agent_on_anomaly`,
`timeout_s` (the hard wall clock for one judgment, fallbacks included), `reasoning_effort` (legacy; `"none"` = off),
`node`, `pi_pkg`, `bridge`.

Validation (`train-snitch check CONFIG`, and at watcher start) refuses: an `api_key` key, anything that looks like a
key, `"!command"` values (Pi would run them), keys inside `base_url`, and unknown or malformed values.

## Examples

Full files in [examples/judge-configs/](../examples/judge-configs) (each extends `examples/generic/local-run.json`).
Only the `judge` block is shown here.

| File | `judge` |
|---|---|
| `codex-luna-default.json` | `{"model": "openai-codex/gpt-6-luna", "credit_cap": 150, "fallbacks": ["openai-codex/gpt-5.6-luna"]}` |
| `anthropic-haiku.json` | `{"model": "anthropic/claude-haiku-4-5", "api_key_env": "ANTHROPIC_API_KEY", "cost_cap_usd": 2}` |
| `google-gemini-flash.json` | `{"model": "google/gemini-3.5-flash", "api_key_env": "GEMINI_API_KEY", "structured_output": "schema", "cost_cap_usd": 1}` |
| `openai-api.json` | `{"model": "openai/gpt-5.4-mini", "api_key_env": "OPENAI_API_KEY", "reasoning": "minimal", "cost_cap_usd": 2}` |
| `openrouter.json` | `{"model": "openrouter/google/gemini-2.5-flash", "api_key_env": "OPENROUTER_API_KEY", "compat": {"openRouterRouting": {"data_collection": "deny"}}}` |
| `groq.json`, `deepseek.json`, `mistral.json`, `xai-grok.json` | built-in providers with their key variable |
| `github-copilot.json` | `{"model": "github-copilot/gpt-5.4-mini", "meter": "none"}` after Pi's `/login github-copilot` |
| `amazon-bedrock.json`, `azure-openai.json`, `google-vertex.json` | cloud providers through their standard credentials |
| `ollama-local.json` | `{"model": "ollama/qwen3:8b", "api": "openai-completions", "base_url": "http://127.0.0.1:11434/v1", "no_auth": true, "compat": {"supportsDeveloperRole": false, "supportsReasoningEffort": false}, "structured_output": "schema"}` |
| `vllm.json`, `llamacpp-server.json`, `lmstudio.json` | other local OpenAI-compatible servers |
| `openai-compatible-gateway.json`, `anthropic-compatible-proxy.json` | your own gateway / proxy, with a cost table so `cost_cap_usd` works |
| `chain-codex-anthropic-ollama.json` | Luna, then Haiku, then a local model |

Check one before running it: `python3 -I bin/train-snitch check examples/judge-configs/ollama-local.json`. It prints
each candidate's endpoint, auth (env var name only) and the equivalent manual `pi` smoke-test command.

## Credentials

Pi resolves auth in this order: a login stored by Pi (`pi`, then `/login`: ChatGPT/Codex, Anthropic, GitHub
Copilot, xAI, ...), the provider's standard env var (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`,
`OPENROUTER_API_KEY`, `GROQ_API_KEY`, `DEEPSEEK_API_KEY`, `MISTRAL_API_KEY`, `XAI_API_KEY`, `AWS_PROFILE`, ...), then
`api_key_env`. A built-in provider needs no `api_key_env` when its standard variable is set; set it anyway to make
the dependency visible. When train-snitch is started by an agent harness, see
[HARNESSES.md](HARNESSES.md#api-keys-and-harness-environments) for getting the key to the watcher.

## How a judgment runs with a model chain

1. Python builds the candidates `[model, *fallbacks]`, drops the ones cooling down (all are sent if every one is),
   and passes them to the bridge with the usual request.
2. For each candidate, the bridge registers its provider config with Pi (memory only), looks up the model, checks
   that Pi has credentials for it, and makes the call:
   - `openai-codex`: the original call (`complete()` with `reasoningEffort`, strict schema, `x-codex-*` headers).
   - any other provider: Pi's portable `completeSimple()` with the thinking level.
   - agent mode (a new rule alert): a Pi `AgentSession` with the same read-only tools, on that model. Non-Codex
     models get the candidate's `max_tokens`, `temperature` and `request_timeout_s` on every turn too.
3. On a failure kind listed in `fallback_on`, it moves to the next candidate, all inside `timeout_s`.
4. The reply carries `model_used` and `attempts` (`judge.jsonl`); alerts carry `judge_model` = the model that
   answered. Every model turn counts toward `max_calls`.
5. Rate limits: a rate-limited model is skipped for `fallback_cooldown_min` (default 15). When every model of the
   chain is rate-limited, the judge pauses until the first cooldown ends (no error counted), except for a chain of
   only `openai-codex` models: a ChatGPT usage limit is account-wide, so, as in 0.1, the judge turns off for the run.
6. The judge also turns off for the run (rules continue) after `max_consecutive_errors` failed judgments, at
   `credit_cap` (Codex), or at `cost_cap_usd`.

### `structured_output`

- `schema`: a strict JSON schema in the request, in the API's own format: `text.format` (OpenAI / Codex / Azure
  Responses), `response_format` (Chat Completions: OpenAI-compatible servers, vLLM, Ollama, llama.cpp), or
  `responseJsonSchema` (Google). Not available for Anthropic, Bedrock and Mistral APIs: the prompt is used there
  (`check` warns when `schema` is set with such an `api`).
- `prompt`: the system prompt asks for JSON and the bridge parses it defensively.
- `auto` (default): `schema` on the Responses APIs, `prompt` elsewhere. Set `schema` for local servers that support
  it (vLLM, Ollama, llama.cpp); small local models follow it much better.

## Limits

- `gpt-6-luna` is in Pi's refreshed model catalog, not the bundled one. On a fresh Pi run `pi update --models` once,
  or keep the `openai-codex/gpt-5.6-luna` fallback.
- Pi cannot reuse a Gemini CLI, Cursor or Claude Code login. Use a Gemini API key / Vertex, OpenRouter, or Pi's own
  Anthropic login.
- `base_url` without `api` only overrides the endpoint of a provider Pi already knows. A new provider name needs
  `api` too (`check` warns), or Pi rejects the registration and the chain stops there.
- Ollama works through its OpenAI-compatible `/v1` endpoint, not its native `/api`. Vendors without an
  OpenAI/Anthropic/Google-compatible API need a Pi extension provider.
- `cost_cap_usd` is only as accurate as Pi's cost table for the model (or your `model_info.cost`). Local servers
  report $0. The Codex credit meter covers only `openai-codex`.
- Verified offline in Docker against Pi 0.80.10: the chain order and failure kinds (missing model, missing key,
  HTTP 429), a fake OpenAI-compatible server (schema in `response_format`, no `reasoning_effort`, key from the named
  env var, never echoed), agent mode through the chain, and the unchanged default request. Not verified live
  against each hosted provider.
