// train-snitch judge bridge: ONE judgment through the Pi harness. Reads a JSON request on stdin, prints ONE JSON line.
//
// mode "single": one stateless call through Pi's model layer (ModelRuntime.complete). No tools, no session.
// mode "agent":  a Pi AgentSession (in-memory, no extensions/skills/context files, no built-in tools) with
//                read-only custom tools over the evidence the watcher already gathered, plus at most one
//                adapter-declared read-only probe command. Hard cap: `max_turns` model turns, then abort.
// Auth is Pi's own (~/.pi/agent/auth.json, refreshed by Pi). This script never prints tokens or auth headers.
//
// Request: {model:"provider/id", mode, max_turns, reasoning_effort, timeout_ms, system_prompt, user_message,
//           data:{logs:{src:[lines]}, series:{name:[...]}, alerts:[...]}, probes:[{name, argv, cwd, arg_pattern,
//           description}], pi_pkg, faux?:[reply texts or {tool, args}]}   (faux = offline test provider)
//          Optional model chain (train_snitch/llm.py; absent = the original single-model path, unchanged):
//           candidates:[{model, reasoning, reasoning_effort?, temperature?, max_tokens, request_timeout_ms,
//                        structured_output:"auto"|"schema"|"prompt", provider_config?}], fallback_on:[kinds]
//           provider_config is a Pi models.json provider object, registered in memory with rt.registerProvider.
// Reply:   {ok, verdict, raw, turns, usage:{input,output,cost}, headers:{x-codex-*}, probes_run:[...], error, ms}
//          + with candidates: model_used, attempts:[{model, ok, error_kind, ms, error?}]
import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";

const req = JSON.parse(readFileSync(0, "utf8"));
const PI = req.pi_pkg || process.env.PI_PKG || "/opt/homebrew/lib/node_modules/@earendil-works/pi-coding-agent";
const t0 = Date.now();
const out = { ok: false, verdict: null, turns: 0, usage: { input: 0, output: 0, cost: 0 }, headers: {}, probes_run: [] };
const done = (extra = {}) => { process.stdout.write(JSON.stringify({ ...out, ...extra, ms: Date.now() - t0 }) + "\n"); process.exit(0); };
const hardStop = setTimeout(() => done({ error: "bridge wall-clock timeout" }), (req.timeout_ms || 150000) + 5000);
hardStop.unref();

const SCHEMA = {
  type: "object", additionalProperties: false,
  required: ["decision", "severity", "topic", "assessment", "what", "evidence", "action", "confidence"],
  properties: {
    decision: { type: "string", enum: ["ok", "watch", "snitch"] },
    severity: { type: "string", enum: ["info", "warn", "critical"] },
    topic: { type: "string" },
    assessment: { type: "string", enum: ["real", "likely_false_alarm", "uncertain", "calm"] },
    what: { type: "string" }, evidence: { type: "array", items: { type: "string" } },
    action: { type: "string" }, confidence: { type: "number" },
  },
};

function parseVerdict(text) {
  if (!text) return null;
  const tries = [text, (text.match(/\{[\s\S]*\}/) || [])[0]];
  for (const s of tries) {
    if (!s) continue;
    try { const v = JSON.parse(s); if (v && typeof v === "object" && (v.decision || v.severity)) return v; } catch {}
  }
  return null;
}
const textOf = (msg) => (msg?.content || []).filter((b) => b.type === "text").map((b) => b.text).join("");
const keepHeaders = (headers) => {
  for (const [k, v] of Object.entries(headers || {})) {
    const key = k.toLowerCase();
    if ((/^x-codex-/.test(key) && key !== "x-codex-turn-state") || key === "retry-after") out.headers[key] = v;
  }
};
const addUsage = (u) => {
  if (!u) return;
  out.usage.input += u.input || 0; out.usage.output += u.output || 0; out.usage.cost += u.cost?.total || 0;
};

let ModelRuntime, piAgent, piAi;
try {
  piAgent = await import(`${PI}/dist/index.js`);
  ModelRuntime = piAgent.ModelRuntime;
  piAi = await import(`${PI}/node_modules/@earendil-works/pi-ai/dist/index.js`);
} catch (e) { done({ error: `cannot load Pi from ${PI}: ${e.message}` }); }

const rt = await ModelRuntime.create({ allowModelNetwork: false });
const maxTurns = Math.max(1, req.max_turns || 3);
let cur = { turns: 0 }, probeUsed = false;
const RESPONSES_APIS = new Set(["openai-codex-responses", "openai-responses", "azure-openai-responses"]);
const GOOGLE_APIS = new Set(["google-generative-ai", "google-vertex"]);
let [provider, ...rest] = String(req.model || "openai-codex/gpt-6-luna").split("/");
let modelId = rest.join("/");
if (Array.isArray(req.faux)) {   // offline test provider, scripted replies
  const faux = piAi.fauxProvider({ provider: "faux-judge", models: [{ id: "faux-judge", reasoning: false }] });
  faux.setResponses(req.faux.map((r) => typeof r === "string"
    ? piAi.fauxAssistantMessage([piAi.fauxText(r)])
    : piAi.fauxAssistantMessage([piAi.fauxToolCall(r.tool, r.args || {})], { stopReason: "toolUse" })));
  rt.registerProvider("faux-judge", {
    api: faux.api, baseUrl: "http://faux.invalid", apiKey: "faux", streamSimple: faux.provider.streamSimple || faux.provider.stream,
    models: [{ id: "faux-judge", name: "faux", reasoning: false, input: ["text"], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 100000, maxTokens: 4000 }],
  });
  if (!Array.isArray(req.candidates)) { provider = "faux-judge"; modelId = "faux-judge"; }
}
if (Array.isArray(req.candidates) && req.candidates.length) await runChain();   // never returns (done() exits)
const model = rt.getModel(provider, modelId);
if (!model) done({ error: `model not found: ${provider}/${modelId}` });
const isCodex = provider === "openai-codex";
const effort = req.reasoning_effort || "none";
const rewrite = (body) => {
  if (!isCodex || !body || typeof body !== "object") return undefined;
  const b = { ...body, text: { ...(body.text || {}), format: { type: "json_schema", name: "verdict", strict: true, schema: SCHEMA } } };
  if (!b.reasoning) b.reasoning = { effort, summary: "auto" };
  return b;
};

// ---------------------------------------------------------------- single: one stateless call
if (req.mode !== "agent") {
  const msg = await rt.complete(model, {
    systemPrompt: req.system_prompt,
    messages: [{ role: "user", content: req.user_message, timestamp: Date.now() }],
  }, {
    reasoningEffort: effort, textVerbosity: "low", transport: "sse", maxRetries: 0, maxTokens: req.max_tokens || 500,
    timeoutMs: 60000, signal: AbortSignal.timeout(req.timeout_ms || 90000),
    onPayload: rewrite, onResponse: ({ headers }) => keepHeaders(headers),
  });
  out.turns = 1; addUsage(msg.usage);
  const raw = textOf(msg);
  const v = parseVerdict(raw);
  done({ ok: !!v, verdict: v, raw: v ? undefined : raw.slice(0, 500), error: msg.stopReason === "error" ? msg.errorMessage : (v ? undefined : "unparseable reply") });
}

// ---------------------------------------------------------------- agent: Pi AgentSession + read-only tools
const { last, aborted } = await runAgent(model, rewrite, "off", req.timeout_ms || 150000);
const raw = textOf(last);
const v = parseVerdict(raw);
done({ ok: !!v, verdict: v, raw: v ? undefined : raw.slice(0, 500),
       error: v ? undefined : (out.error || (aborted ? `turn cap ${maxTurns} reached` : (last?.stopReason === "error" ? last.errorMessage : "unparseable reply"))) });

// ---------------------------------------------------------------- shared by both paths
function agentTools() {
  const { defineTool } = piAgent;
  const data = req.data || {};
  const S = (props, required = []) => ({ type: "object", properties: props, required, additionalProperties: false });
  // tool calls are allowed in turns 1..maxTurns-1; the last allowed tool result tells the model to answer now
  const lastToolTurn = () => cur.turns + 1 >= maxTurns - 1;
  const txt = (s) => ({ content: [{ type: "text", text: String(s).slice(0, 6000)
    + (lastToolTurn() ? "\n[tool budget used: reply now with ONLY the JSON verdict]" : "") }], details: {} });
  const tools = [
    defineTool({
      name: "read_log", label: "read_log",
      description: "Recent lines of one log source the watcher already collected (read-only). Sources: " + Object.keys(data.logs || {}).join(", "),
      parameters: S({ source: { type: "string" }, last: { type: "number" }, grep: { type: "string" } }, ["source"]),
      execute: async (_id, p) => {
        let lines = (data.logs || {})[p.source];
        if (!lines) return txt(`no such source; have: ${Object.keys(data.logs || {}).join(", ")}`);
        if (p.grep) { try { const r = new RegExp(p.grep, "i"); lines = lines.filter((l) => r.test(l)); } catch { lines = lines.filter((l) => l.includes(p.grep)); } }
        return txt(lines.slice(-Math.min(Math.max(p.last || 40, 1), 120)).join("\n") || "(no lines)");
      },
    }),
    defineTool({
      name: "get_series", label: "get_series",
      description: "A metric series or table the watcher keeps (read-only). Names: " + Object.keys(data.series || {}).join(", "),
      parameters: S({ name: { type: "string" }, last: { type: "number" } }, ["name"]),
      execute: async (_id, p) => {
        const s = (data.series || {})[p.name];
        if (s === undefined) return txt(`no such series; have: ${Object.keys(data.series || {}).join(", ")}`);
        return txt(JSON.stringify(Array.isArray(s) ? s.slice(-Math.min(p.last || 30, 200)) : s));
      },
    }),
    defineTool({
      name: "alert_history", label: "alert_history", description: "The watcher's recent alert records (read-only).",
      parameters: S({ last: { type: "number" } }),
      execute: async (_id, p) => txt(JSON.stringify((data.alerts || []).slice(-Math.min(p.last || 15, 50)))),
    }),
  ];
  const probes = req.probes || [];
  if (probes.length) {
    tools.push(defineTool({
      name: "run_probe", label: "run_probe",
      description: "Run ONE read-only diagnostic command (at most once per judgment). " + probes.map((p) => `${p.name}: ${p.description}`).join(" | "),
      parameters: S({ name: { type: "string" }, arg: { type: "string" } }, ["name"]),
      execute: async (_id, p) => {
        const pr = probes.find((x) => x.name === p.name);
        if (!pr) return txt("unknown probe");
        if (probeUsed) return txt("probe budget used (one per judgment)");
        const arg = p.arg || "";
        if (pr.arg_pattern && !new RegExp(pr.arg_pattern).test(arg)) return txt(`arg must match ${pr.arg_pattern}`);
        probeUsed = true; out.probes_run.push(`${pr.name} ${arg}`.trim());
        const argv = pr.argv.map((a) => a.replace("{arg}", arg));
        const r = spawnSync(argv[0], argv.slice(1), { cwd: pr.cwd || undefined, env: { ...process.env, ...(pr.env || {}) }, timeout: 60000, encoding: "utf8", maxBuffer: 1 << 20 });
        return txt(`rc ${r.status}\n${(r.stdout || "").slice(-4000)}\n${(r.stderr || "").slice(-800)}`);
      },
    }));
  }
  return tools;
}

// one AgentSession run (per model); turns are counted per run in `cur` and in total in out.turns
// streamOpts (chain, non-Codex): maxTokens / temperature / timeoutMs merged into every turn's Pi stream options
async function runAgent(model, rewriteFn, thinkingLevel, timeoutMs, streamOpts) {
  const { createAgentSession, DefaultResourceLoader, SessionManager, SettingsManager } = piAgent;
  cur = { turns: 0 };
  out.error = undefined;
  const tools = agentTools();
  const cwd = process.cwd();
  const settingsManager = SettingsManager.inMemory({ compaction: { enabled: false }, retry: { enabled: false }, transport: "sse" });
  const resourceLoader = new DefaultResourceLoader({
    cwd, agentDir: process.env.PI_AGENT_DIR || process.env.PI_CODING_AGENT_DIR || `${process.env.HOME}/.pi/agent`, settingsManager,
    noExtensions: true, noSkills: true, noPromptTemplates: true, noThemes: true, noContextFiles: true,
    systemPromptOverride: () => req.system_prompt,
    extensionFactories: [(pi) => {
      pi.on("before_provider_request", (event) => rewriteFn(event.payload));
      pi.on("after_provider_response", (event) => keepHeaders(event.headers));
    }],
  });
  await resourceLoader.reload();
  const { session } = await createAgentSession({
    cwd, model, thinkingLevel, modelRuntime: rt, resourceLoader, settingsManager,
    sessionManager: SessionManager.inMemory(cwd), noTools: "builtin", customTools: tools, tools: tools.map((t) => t.name),
  });
  if (streamOpts && session.agent && typeof session.agent.streamFn === "function") {
    const base = session.agent.streamFn;
    session.agent.streamFn = (m, context, options) => base(m, context, { ...options, ...streamOpts });
  }
  let last = null, aborted = false;
  session.subscribe((ev) => {
    if (ev.type === "turn_end") {
      out.turns += 1; cur.turns += 1; addUsage(ev.message?.usage); last = ev.message;
      const wantsMore = (ev.message?.content || []).some((b) => b.type === "toolCall");
      if (wantsMore && cur.turns >= maxTurns) { aborted = true; session.abort().catch(() => {}); }
    }
  });
  try {
    await Promise.race([
      session.prompt(req.user_message),
      new Promise((_, rej) => setTimeout(() => rej(new Error("judge timeout")), timeoutMs)),
    ]);
  } catch (e) {
    out.error = String(e.message || e);
    try { await session.abort(); } catch {}
  }
  session.dispose?.();
  return { last, aborted };
}

// ---------------------------------------------------------------- model chain (train_snitch/llm.py)
// Try each candidate in order inside the one wall-clock budget; move on when the failure kind is in fallback_on.
async function runChain() {
  const fallbackOn = new Set(req.fallback_on || ["not_found", "no_auth", "error", "rate_limit", "timeout"]);
  const deadline = t0 + (req.timeout_ms || 90000);
  const attempts = [];
  let last = { kind: "error", error: "no candidates" };
  for (const c of req.candidates) {
    const left = deadline - Date.now();
    if (left < 1500) { last = { kind: "timeout", error: `no time left for ${c.model}` }; attempts.push({ model: c.model, ok: false, error_kind: "timeout", ms: 0 }); break; }
    const ta = Date.now();
    let r;
    try { r = await tryCandidate(c, left); } catch (e) { r = { kind: "error", error: String(e?.message || e) }; }
    attempts.push({ model: c.model, ok: !!r.verdict, error_kind: r.verdict ? undefined : r.kind, ms: Date.now() - ta,
                    ...(r.verdict ? {} : { error: String(r.error || "").slice(0, 200) }) });
    if (r.verdict) done({ ok: true, verdict: r.verdict, model_used: c.model, attempts, error: undefined });
    last = r;
    if (!fallbackOn.has(r.kind)) break;
  }
  done({ ok: false, verdict: null, raw: last.raw, model_used: null, attempts,
         error: attempts.length > 1 ? `all ${attempts.length} judge models failed; last ${attempts.at(-1).model}: ${last.error}` : last.error });
}

function kindOf(msg, stopReason) {
  const m = String(msg || "");
  // Pi words a Codex 429 as "You have hit your ChatGPT usage limit (... plan). Try again in ~N min."
  if (/usage.?limit|rate.?limit|\b429\b|insufficient|quota|credits? (?:exhausted|depleted)/i.test(m)) return "rate_limit";
  if (/\b40[13]\b|unauthori[sz]ed|forbidden|api key|no credentials|authentication|invalid.*key/i.test(m)) return "no_auth";
  if (stopReason === "aborted" || /time(?:d)? ?out|abort/i.test(m)) return "timeout";
  return "error";
}

async function tryCandidate(c, leftMs) {
  const [prov, ...rest] = String(c.model || "").split("/");
  const mid = rest.join("/");
  if (c.provider_config) {
    try { rt.registerProvider(prov, c.provider_config); }
    catch (e) { return { kind: "config", error: `Pi rejected the provider config for ${prov}: ${e.message}` }; }
  }
  const m = rt.getModel(prov, mid);
  if (!m) return { kind: "not_found", error: `model not found: ${c.model} (check \`pi --list-models\`; run \`pi update --models\` for new ids)` };
  let auth;
  try { auth = await rt.getAuth(m); } catch { auth = undefined; }
  if (!auth) return { kind: "no_auth", error: `no credentials for ${prov}: log in with Pi, export its env var, or set judge.api_key_env` };
  const codex = prov === "openai-codex";
  const level = c.reasoning || "off";
  const effort = c.reasoning_effort || (level === "off" ? "none" : (m.thinkingLevelMap?.[level] ?? level));
  const so = c.structured_output === "auto" || !c.structured_output
    ? (RESPONSES_APIS.has(m.api) ? "schema" : "prompt") : c.structured_output;
  const rewriteC = (body) => {
    if (!body || typeof body !== "object") return undefined;
    let b = body;
    if (so === "schema") {
      if (RESPONSES_APIS.has(m.api)) b = { ...b, text: { ...(b.text || {}), format: { type: "json_schema", name: "verdict", strict: true, schema: SCHEMA } } };
      else if (m.api === "openai-completions") b = { ...b, response_format: { type: "json_schema", json_schema: { name: "verdict", strict: true, schema: SCHEMA } } };
      else if (GOOGLE_APIS.has(m.api)) b = { ...b, config: { ...(b.config || {}), responseMimeType: "application/json", responseJsonSchema: SCHEMA } };
    }
    if (codex && !b.reasoning) b = { ...b, reasoning: { effort, summary: "auto" } };
    return b === body ? undefined : b;
  };
  const signal = AbortSignal.timeout(leftMs);
  if (req.mode !== "agent") {
    const ctx = { systemPrompt: req.system_prompt, messages: [{ role: "user", content: req.user_message, timestamp: Date.now() }] };
    const common = { transport: "sse", maxRetries: 0, maxTokens: c.max_tokens || req.max_tokens || 500,
                     timeoutMs: c.request_timeout_ms || 60000, signal, onPayload: rewriteC, onResponse: ({ headers }) => keepHeaders(headers) };
    if (c.temperature !== undefined && c.temperature !== null) common.temperature = c.temperature;
    // Codex: the exact original call (complete + reasoningEffort). Others: Pi's portable completeSimple + thinking level.
    const msg = codex
      ? await rt.complete(m, ctx, { ...common, reasoningEffort: effort, textVerbosity: "low" })
      : await rt.completeSimple(m, ctx, { ...common, ...(level !== "off" ? { reasoning: level } : {}) });
    out.turns += 1; addUsage(msg.usage);
    const raw = textOf(msg);
    const v = parseVerdict(raw);
    if (v) return { verdict: v };
    if (msg.stopReason === "error" || msg.stopReason === "aborted") return { kind: kindOf(msg.errorMessage, msg.stopReason), error: msg.errorMessage || msg.stopReason };
    return { kind: "unparseable", error: "unparseable reply", raw: raw.slice(0, 500) };
  }
  // Codex keeps its original agent request; other providers get the candidate's limits on every turn
  const streamOpts = codex ? undefined : {
    maxTokens: c.max_tokens || req.max_tokens || 500, timeoutMs: c.request_timeout_ms || 60000,
    ...(c.temperature !== undefined && c.temperature !== null ? { temperature: c.temperature } : {}),
  };
  const { last, aborted } = await runAgent(m, rewriteC, level, leftMs, streamOpts);
  const raw = textOf(last);
  const v = parseVerdict(raw);
  if (v) return { verdict: v };
  if (aborted) return { kind: "turn_cap", error: `turn cap ${maxTurns} reached` };
  if (out.error) return { kind: kindOf(out.error, "aborted"), error: out.error };
  if (last?.stopReason === "error" || last?.stopReason === "aborted") return { kind: kindOf(last.errorMessage, last.stopReason), error: last.errorMessage || last.stopReason };
  return { kind: "unparseable", error: "unparseable reply", raw: raw.slice(0, 500) };
}
