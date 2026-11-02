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
// Reply:   {ok, verdict, raw, turns, usage:{input,output,cost}, headers:{x-codex-*}, probes_run:[...], error, ms}
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
  provider = "faux-judge"; modelId = "faux-judge";
}
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
const { createAgentSession, DefaultResourceLoader, SessionManager, SettingsManager, defineTool } = piAgent;
const data = req.data || {};
const S = (props, required = []) => ({ type: "object", properties: props, required, additionalProperties: false });
const maxTurns = Math.max(1, req.max_turns || 3);
// tool calls are allowed in turns 1..maxTurns-1; the last allowed tool result tells the model to answer now
const lastToolTurn = () => out.turns + 1 >= maxTurns - 1;
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
let probeUsed = false;
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

const cwd = process.cwd();
const settingsManager = SettingsManager.inMemory({ compaction: { enabled: false }, retry: { enabled: false }, transport: "sse" });
const resourceLoader = new DefaultResourceLoader({
  cwd, agentDir: process.env.PI_AGENT_DIR || `${process.env.HOME}/.pi/agent`, settingsManager,
  noExtensions: true, noSkills: true, noPromptTemplates: true, noThemes: true, noContextFiles: true,
  systemPromptOverride: () => req.system_prompt,
  extensionFactories: [(pi) => {
    pi.on("before_provider_request", (event) => rewrite(event.payload));
    pi.on("after_provider_response", (event) => keepHeaders(event.headers));
  }],
});
await resourceLoader.reload();
const { session } = await createAgentSession({
  cwd, model, thinkingLevel: "off", modelRuntime: rt, resourceLoader, settingsManager,
  sessionManager: SessionManager.inMemory(cwd), noTools: "builtin", customTools: tools, tools: tools.map((t) => t.name),
});
let last = null, aborted = false;
session.subscribe((ev) => {
  if (ev.type === "turn_end") {
    out.turns += 1; addUsage(ev.message?.usage); last = ev.message;
    const wantsMore = (ev.message?.content || []).some((b) => b.type === "toolCall");
    if (wantsMore && out.turns >= maxTurns) { aborted = true; session.abort().catch(() => {}); }
  }
});
try {
  await Promise.race([
    session.prompt(req.user_message),
    new Promise((_, rej) => setTimeout(() => rej(new Error("judge timeout")), req.timeout_ms || 150000)),
  ]);
} catch (e) {
  out.error = String(e.message || e);
  try { await session.abort(); } catch {}
}
const raw = textOf(last);
const v = parseVerdict(raw);
session.dispose?.();
done({ ok: !!v, verdict: v, raw: v ? undefined : raw.slice(0, 500),
       error: v ? undefined : (out.error || (aborted ? `turn cap ${maxTurns} reached` : (last?.stopReason === "error" ? last.errorMessage : "unparseable reply"))) });
