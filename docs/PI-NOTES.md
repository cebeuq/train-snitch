# Pi 0.80.10 as the judge backend: notes

> Update, Oct 31 8:20 PM: the budget below ("60 calls, 1 per 15 min") was replaced. The judge now runs every
> 60 s by default, with a per-run Codex credit cap (default 150) and back-off to rules-only after 3 failed calls in
> a row; see README "The judge". Two live calls through `judge/pi_judge.mjs` (single + agent mode) confirmed
> the strict schema and the AgentSession path: about 900-1,200 tokens in, about 120 out, 2-3 s each.

Researched Sat Oct 31, 2026, 7:30 PM. Sources: the installed package at
`/opt/homebrew/lib/node_modules/@earendil-works/pi-coding-agent` (`docs/sdk.md`,
`docs/rpc.md`, `docs/json.md`, `docs/extensions.md`, `docs/providers.md`,
`node_modules/@earendil-works/pi-ai/README.md`, and the `dist/` source).

## TL;DR (recommended design)

- Pi has 3 layers. The judge needs only the bottom one:
  1. **pi-ai `Models`** (`ModelRuntime` in coding-agent): one-shot `complete()` over any provider, with Pi's
     auth (`~/.pi/agent/auth.json`, OAuth auto-refresh under a file lock). **No agent loop, no tools, no session.**
  2. `AgentSession` (SDK `createAgentSession`): the full agent loop (tools, compaction, sessions).
  3. CLI modes: `pi -p` (print), `--mode json` (event JSONL), `--mode rpc` (stdin/stdout JSON protocol).
- **Use layer 1 for every judge call**: a new `Context` with one system prompt and one user message → `complete()`.
  Stateless by construction (`store:false`, no history, no prompt-cache key). Proven live below.
- Use layer 2 only when the judge should call read-only tools itself (calm minutes don't: the deterministic layer
  gathers evidence, Luna only judges). Layer 3 is the fallback for a non-Node caller.
- The core imports Pi from wherever it is installed (`PI_PKG` env, default the Homebrew global path). No
  project `node_modules` needed. If you pin Pi as a dependency, keep it
  project-local with a lockfile.

## Proof: one live Luna call

A single stateless call with `openai-codex/gpt-6-luna`, `reasoningEffort:"none"`, `transport:"sse"`, `maxRetries:0`
and a strict `json_schema` injected through `onPayload` returned valid JSON in 1.6 s (102 tokens in, 54 out, no
reasoning tokens). The credit balance, read by the free meter below, dropped by about 0.0009 credits about a
minute after the call (the debit lags).

At this size, a per-run call cap is far below any sensible credit cutoff. The guard is still worth keeping in case
of bigger prompts.

## Reading the Codex meter and credits (free, no model call)

Two ways, both verified:

1. **`codex app-server`, method `account/rateLimits/read`** → `train_snitch/meter.py` (stdlib only).
   It returns `rateLimits.primary.usedPercent/resetsAt`, `credits.balance/hasCredits`, `rateLimitReachedType`,
   `planType`, and `ordinaryUsageAllowed`. It uses the Codex CLI's own login. Pi's `openai-codex` login is the
   **same account** (accountId matches), so this meter covers Pi's calls. Use it at start, every N calls,
   and at end. The balance lags about 1 minute behind a call.
2. **Response headers on every call** (SSE transport only; `onResponse` / extension `after_provider_response`):
   `x-codex-primary-used-percent`, `x-codex-primary-reset-at`, `x-codex-credits-balance`,
   `x-codex-credits-has-credits`, `x-codex-active-limit`, `x-codex-plan-type`. These show the balance
   **before** this call is debited. Drop `x-codex-turn-state`, an opaque blob.

Credit guard: `start = meter()`. After each call, read the header balance (or `meter()` every ~10 calls).
If `start - balance > 50`, switch to rules-only.

## Minimal judge call (layer 1)

```js
const { ModelRuntime } = await import(`${PI_PKG}/dist/index.js`);
const rt = await ModelRuntime.create({ allowModelNetwork: false }); // no catalog fetch at startup
const model = rt.getModel("openai-codex", "gpt-6-luna");
const msg = await rt.complete(model, { systemPrompt, messages: [{ role: "user", content, timestamp: Date.now() }] }, {
  reasoningEffort: "none", textVerbosity: "low", transport: "sse", maxRetries: 0,
  timeoutMs: 60_000, signal: AbortSignal.timeout(90_000), maxTokens: 400,
  onPayload: (b) => ({ ...b, text: { ...b.text, format: { type: "json_schema", name: "verdict", strict: true, schema } } }),
  onResponse: ({ status, headers }) => { /* read x-codex-* */ },
});
// msg.stopReason: "stop" | "length" | "error" | "aborted"; msg.errorMessage; msg.usage {input,output,reasoning,totalTokens,cost}
// text = msg.content.filter(b => b.type === "text").map(b => b.text).join("")
```

Gotchas found in the source:

- `completeSimple(..., { reasoning: "off" })` **omits** the reasoning field, so the server default effort
  is used. That is not "none". Use `complete()` + `reasoningEffort: "none"` (Luna maps it to `none`), or
  `reasoning: "minimal"` (maps to `low`).
- Errors do not throw. They come back as `stopReason:"error"` + `errorMessage`. Treat 429
  `usage_limit_reached` / `rate_limit_exceeded` as "go rules-only".
- `maxRetries` is 0 by default in the Codex transport. Pass 0 explicitly anyway. Retries happen in the
  AgentSession layer (`settings.retry`), not in `complete()`, so layer 1 never silently repeats a call.
- `timeoutMs` covers the wait for response headers on SSE, or socket idle on WebSocket. Add an outer
  `AbortSignal.timeout()` for a hard wall-clock cap.
- `transport:"auto"` (the default) prefers WebSocket. `onResponse` headers only fire on SSE, so use `"sse"`.
- Do not pass `sessionId` (it becomes `prompt_cache_key` and reuses a WebSocket). Calls stay stateless anyway.
- `text.verbosity` is already `low` by default on Codex.
- Structured output alternatives: (a) `json_schema` via `onPayload`, used above and works on Codex;
  (b) one TypeBox tool in `context.tools` + `toolChoice:"required"`, then read `toolCall.arguments`.
  This is portable to Anthropic. (c) Prompt for JSON, then `JSON.parse` with a regex fallback.
  The core should parse defensively whichever way is used.

## Model selection

- Luna: `openai-codex/gpt-6-luna` (also `gpt-5.6-luna`). Context 272K, reasoning levels
  off→none, minimal→low, low, medium, high, xhigh, max. Auth: Pi's Codex OAuth (log in first).
- CLI syntax: `--model openai-codex/gpt-6-luna:low` (the `:thinking` suffix is optional).
- **Claude Haiku as an alternative backend**: provider `anthropic`. It is **not usable right now**: Pi has
  no Anthropic auth (`auth.json` only has `openai-codex`), and the built-in catalog only lists
  `claude-haiku-4-5`. To enable it:
  - set `ANTHROPIC_API_KEY`, or run `/login anthropic` in interactive Pi. Pi's docs say subscription OAuth
    in third-party harnesses bills per token as extra usage.
  - then run `pi update --models` to refresh catalogs, or add the model id to `~/.pi/agent/models.json`.
    The exact Haiku 5.5 id is unverified; check `pi --list-models haiku` once auth exists.
  - In code, only `getModel("anthropic", id)` changes. Use the tool-call structured-output path (b);
    `json_schema` injection is Codex/OpenAI-specific. Skip the meter (it is Codex-only); use `msg.usage.cost`.
- Make the backend a config value (`provider/model`) in the generic core. Keep the meter adapter pluggable.

## Other headless options (not used for the judge)

- **Print mode, any language**:
  `pi -p --mode json --no-session -nt -nc -ne -ns -np --offline --model openai-codex/gpt-6-luna:minimal --system-prompt "<judge prompt>" "<evidence>"`
  emits event JSONL. Take the last `message_end` (assistant) → `.message.content[].text`, `.message.usage`.
  `-nc` matters: without it `~/.pi/agent/AGENTS.md` and project AGENTS.md/CLAUDE.md are appended to the
  system prompt, which costs tokens and leaks local agent instructions. Even with `--system-prompt`, Pi appends
  `Current working directory: ...`. There is no CLI flag for json_schema, retries, or timeout; for those,
  load an extension with `-e ext.ts` that rewrites the payload (`before_provider_request`) and reads
  headers (`after_provider_response`). Cost: about 0.5 s Node startup per call. Flags were checked from
  `--help` and the source; not run live, to save a call.
- **RPC mode**: `pi --mode rpc --no-session ...`, JSONL over stdin/stdout (`prompt`, `abort`, `new_session`,
  `get_session_stats` → tokens + cost). Split records on `\n` only (Node `readline` is not compliant).
  For a judge, call `new_session` before each prompt or history accumulates.
- **SDK agent loop**: `createAgentSession({ sessionManager: SessionManager.inMemory(), noTools:"all"` or
  `customTools:[...], resourceLoader: new DefaultResourceLoader({ systemPromptOverride, noExtensions... }),
  settingsManager: SettingsManager.inMemory({ compaction:{enabled:false}, retry:{enabled:false} }) })`.
  Usage then comes from `turn_end.message.usage`. Only worth it if the judge gets tools.
- **Extensions** (`pi.registerTool`, `pi.on("before_provider_request"|"after_provider_response"|...)`) are
  the Pi-native way to add tools and hooks to the CLI/RPC modes. They are not needed for layer 1.

## Keeping each judge call cheap and stateless

1. Fresh one-message `Context` per call; never reuse `msg` in a later context.
2. Small fixed system prompt (rubric + schema); send evidence as a compact digest (top-N log lines,
   last-K metric points, rule hits). Cap user content at about 4-8K chars.
3. `reasoningEffort:"none"` by default; `"low"` only for critical-severity confirmation.
4. `maxTokens` around 400, `maxRetries:0`, hard timeout 90 s. On error or timeout, the rules-layer verdict stands.
5. Budget: at most 60 calls per run, at most 1 per 15 min when calm, and calls only on rule-layer state
   changes. Meter credits at start, every 10 calls, and at end (`meter.py`), plus headers per call.
6. No tools, no session file (`ModelRuntime` writes nothing). Never log `auth.json` contents or auth headers.
