"""Judge LLM config: which model(s) the Pi judge uses, validated and mapped onto Pi's own provider config.

No API clients live here. Every option maps onto something Pi 0.80.x already understands:
- `model` "provider/id"            -> ModelRuntime.getModel(provider, id) (same string as `pi --model`)
- `api`, `base_url`, `api_key_env`, `no_auth`, `headers`, `compat`, `model_info`
                                    -> an in-memory ModelRuntime.registerProvider(provider, cfg), the same object
                                       shape as a provider entry in Pi's models.json (nothing is written to disk)
- `reasoning`                       -> Pi thinking level (off|minimal|low|medium|high|xhigh|max)
- `temperature`, `max_tokens`, `request_timeout_s` -> Pi StreamOptions
- `fallbacks`                       -> more candidates, tried in order by the bridge inside one judgment

API keys never appear in a config: `api_key_env` names an environment variable and Pi resolves "$NAME" itself.
Pi also runs "!command" values in apiKey/headers; this module refuses them, so a config can never make Pi run a
command. With none of the new keys set, the bridge request is exactly the one train-snitch 0.1 sent.
"""
from __future__ import annotations

import re

THINKING = ("off", "minimal", "low", "medium", "high", "xhigh", "max")
APIS = ("openai-completions", "openai-responses", "anthropic-messages", "google-generative-ai")
STRUCTURED = ("auto", "schema", "prompt")
FALLBACK_KINDS = ("not_found", "no_auth", "error", "rate_limit", "timeout", "unparseable", "config")
DEFAULT_FALLBACK_ON = ["not_found", "no_auth", "error", "rate_limit", "timeout"]
METERS = ("auto", "codex", "none")
CODEX = "openai-codex"

# keys a candidate (the primary judge block or one fallback object) may carry
CANDIDATE_KEYS = ("model", "api", "base_url", "api_key_env", "no_auth", "headers", "compat", "model_info",
                  "reasoning", "temperature", "max_tokens", "request_timeout_s", "structured_output")
# judge keys added in 0.2 that change the bridge request; with none of them set the request is the 0.1 one
REQUEST_KEYS = ("api", "base_url", "api_key_env", "no_auth", "headers", "compat", "model_info", "reasoning",
                "temperature", "request_timeout_s", "structured_output", "fallbacks", "fallback_on")
MODEL_INFO_KEYS = {"context_window": "contextWindow", "max_tokens": "maxTokens", "reasoning": "reasoning",
                   "input": "input", "cost": "cost", "name": "name"}
ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
ENV_REF = re.compile(r"^\$(?:\{[A-Za-z_][A-Za-z0-9_]*\}|[A-Za-z_][A-Za-z0-9_]*)$")
SECRETISH = re.compile(r"key|token|secret|auth|password|bearer", re.I)
LOOKS_LIKE_KEY = re.compile(r"^(?:sk-|sk_|hf_|gsk_|xai-|AKIA|AIza|ghp_|github_pat_|Bearer\s)", re.I)


def uses_candidates(j: dict) -> bool:
    """The bridge's model-chain request: any 0.2 key set, or a non-Codex model (the 0.1 single-model path sends
    Codex-only options). The default Codex config keeps the 0.1 request byte for byte."""
    if split_model(j.get("model") or "openai-codex/")[0] != CODEX:
        return True
    return any(j.get(k) is not None and j.get(k) is not False and j.get(k) != [] and j.get(k) != {}
               for k in REQUEST_KEYS)


def split_model(model) -> tuple[str, str]:
    provider, _, mid = str(model or "").partition("/")
    return provider, mid


# ---------------------------------------------------------------- validation
def validate(j: dict) -> tuple[list[str], list[str]]:
    """(errors, warnings) for a merged `judge` block. Errors stop the watcher at start; nothing here calls a model."""
    errs, warns = [], []
    if not isinstance(j, dict):
        return ["judge must be an object"], []
    if "api_key" in j:
        errs.append("judge.api_key is not allowed (keys never go in a config): set judge.api_key_env to the NAME of "
                    "an environment variable, or log in with Pi (`pi`, then /login)")
    _check_candidate(j, "judge", errs, warns, primary=True)
    fb = j.get("fallbacks")
    if fb is not None:
        if not isinstance(fb, list):
            errs.append("judge.fallbacks must be a list of \"provider/id\" strings or objects")
        else:
            for i, c in enumerate(fb):
                where = f"judge.fallbacks[{i}]"
                if isinstance(c, str):
                    _check_candidate({"model": c}, where, errs, warns)
                elif isinstance(c, dict):
                    extra = sorted(set(c) - set(CANDIDATE_KEYS))
                    if extra:
                        errs.append(f"{where}: unknown keys {extra} (allowed: {', '.join(CANDIDATE_KEYS)})")
                    if "api_key" in c:
                        errs.append(f"{where}.api_key is not allowed: use api_key_env")
                    _check_candidate(c, where, errs, warns)
                else:
                    errs.append(f"{where} must be a \"provider/id\" string or an object")
    fo = j.get("fallback_on")
    if fo is not None and (not isinstance(fo, list) or any(x not in FALLBACK_KINDS for x in fo)):
        errs.append(f"judge.fallback_on must be a list drawn from {', '.join(FALLBACK_KINDS)}")
    for k in ("fallback_cooldown_min", "cost_cap_usd", "credit_cap", "every_s", "timeout_s"):
        v = j.get(k)
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0):
            errs.append(f"judge.{k} must be a non-negative number")
    if j.get("meter", "auto") not in METERS:
        errs.append(f"judge.meter must be one of {', '.join(METERS)}")
    if j.get("pi_agent_dir") is not None and not isinstance(j.get("pi_agent_dir"), str):
        errs.append("judge.pi_agent_dir must be a path string")
    if not errs and j.get("enabled") and meter_kind(j) == "none" and j.get("cost_cap_usd") is None and not _all_local(j):
        warns.append("no spend guard: the Codex credit meter only covers openai-codex; set judge.cost_cap_usd "
                     "(uses Pi's own per-call cost figure) to cap a paid API")
    return errs, warns


def _check_candidate(c: dict, where: str, errs: list, warns: list, primary: bool = False) -> None:
    m = c.get("model")
    provider, mid = split_model(m)
    if not isinstance(m, str) or not provider or not mid:
        errs.append(f"{where}.model must be \"provider/id\" as Pi knows it (e.g. openai-codex/gpt-6-luna, "
                    f"anthropic/claude-haiku-4-5, ollama/qwen3:8b); got {m!r}")
    api = c.get("api")
    if api is not None and api not in APIS:
        errs.append(f"{where}.api must be one of {', '.join(APIS)}")
    if api and not c.get("base_url"):
        errs.append(f"{where}: api is set, so base_url is required (a custom provider needs an endpoint)")
    bu = c.get("base_url")
    if bu is not None and (not isinstance(bu, str) or not re.match(r"^https?://", bu)):
        errs.append(f"{where}.base_url must be an http(s) URL")
    elif isinstance(bu, str) and re.search(r"[?&](?:key|api_key|token)=|//[^/@]+:[^/@]+@", bu, re.I):
        errs.append(f"{where}.base_url must not carry a key; use api_key_env or headers with \"$ENV\"")
    ke = c.get("api_key_env")
    if ke is not None:
        if not isinstance(ke, str) or not ENV_NAME.match(ke):
            errs.append(f"{where}.api_key_env must be the NAME of an environment variable (like OPENROUTER_API_KEY), "
                        "not a key or a command")
    if c.get("no_auth") not in (None, True, False):
        errs.append(f"{where}.no_auth must be true or false")
    if c.get("no_auth") and ke:
        errs.append(f"{where}: no_auth and api_key_env are mutually exclusive")
    h = c.get("headers")
    if h is not None:
        if not isinstance(h, dict):
            errs.append(f"{where}.headers must be an object")
        else:
            for k, v in h.items():
                if not isinstance(v, str):
                    errs.append(f"{where}.headers.{k} must be a string")
                elif v.startswith("!"):
                    errs.append(f"{where}.headers.{k}: \"!command\" values are refused (train-snitch never lets a "
                                "config run commands through Pi)")
                elif not ENV_REF.match(v) and (SECRETISH.search(k) or LOOKS_LIKE_KEY.match(v)):
                    errs.append(f"{where}.headers.{k} looks like a secret: write \"$ENV_NAME\" and export the value")
    for k in ("compat",):
        if c.get(k) is not None and not isinstance(c.get(k), dict):
            errs.append(f"{where}.{k} must be an object (passed to Pi unchanged)")
    mi = c.get("model_info")
    if mi is not None:
        if not isinstance(mi, dict):
            errs.append(f"{where}.model_info must be an object")
        else:
            bad = sorted(set(mi) - set(MODEL_INFO_KEYS))
            if bad:
                errs.append(f"{where}.model_info: unknown keys {bad} (allowed: {', '.join(MODEL_INFO_KEYS)})")
    r = c.get("reasoning")
    if r is not None and r not in THINKING:
        errs.append(f"{where}.reasoning must be one of {', '.join(THINKING)}")
    t = c.get("temperature")
    if t is not None and (isinstance(t, bool) or not isinstance(t, (int, float)) or not 0 <= t <= 2):
        errs.append(f"{where}.temperature must be a number from 0 to 2")
    for k in ("max_tokens", "request_timeout_s"):
        v = c.get(k)
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0):
            errs.append(f"{where}.{k} must be a positive number")
    so = c.get("structured_output")
    if so is not None and so not in STRUCTURED:
        errs.append(f"{where}.structured_output must be one of {', '.join(STRUCTURED)}")
    for k, v in c.items():
        if isinstance(v, str) and LOOKS_LIKE_KEY.match(v) and k not in ("model",):
            errs.append(f"{where}.{k} looks like an API key; keys never go in a config (use api_key_env)")


def _all_local(j: dict) -> bool:
    cands = candidates(j)
    return bool(cands) and all(c.get("no_auth") or re.match(r"^https?://(?:127\.0\.0\.1|localhost|\[::1\])[:/]",
                                                             str(c.get("base_url") or "")) for c in cands)


# ---------------------------------------------------------------- mapping onto Pi
def candidates(j: dict) -> list[dict]:
    """[primary, *fallbacks] as plain candidate dicts. A fallback string or object inherits the primary's
    reasoning / temperature / limits / structured_output, but never its endpoint or auth (those are per provider)."""
    inherit = {k: j.get(k) for k in ("reasoning", "temperature", "max_tokens", "request_timeout_s", "structured_output")}
    prim = {k: j.get(k) for k in CANDIDATE_KEYS}
    out = [prim]
    for c in j.get("fallbacks") or []:
        c = {"model": c} if isinstance(c, str) else dict(c)
        out.append({**inherit, **{k: v for k, v in c.items() if v is not None}})
    return out


def provider_config(c: dict) -> dict | None:
    """The object passed to Pi's ModelRuntime.registerProvider(provider, cfg) (models.json provider shape), or None
    when the candidate uses Pi's built-in provider exactly as configured (catalog, auth.json, env)."""
    if not any(c.get(k) for k in ("api", "base_url", "api_key_env", "no_auth", "headers", "compat", "model_info")):
        return None
    cfg: dict = {}
    if c.get("base_url"):
        cfg["baseUrl"] = c["base_url"]
    if c.get("api"):
        cfg["api"] = c["api"]
    if c.get("api_key_env"):
        cfg["apiKey"] = "$" + c["api_key_env"]
    elif c.get("no_auth"):
        cfg["apiKey"] = "local"        # keyless local server: Pi still wants a value (its docs advise a dummy)
    if c.get("headers"):
        cfg["headers"] = dict(c["headers"])
    if c.get("compat"):
        cfg["compat"] = dict(c["compat"])
    if c.get("api") or c.get("model_info"):
        _, mid = split_model(c.get("model"))
        mi = c.get("model_info") or {}
        m = {"id": mid, "name": mi.get("name") or mid, "reasoning": bool(mi.get("reasoning", False)),
             "input": list(mi.get("input") or ["text"]), "contextWindow": int(mi.get("context_window") or 128000),
             "maxTokens": int(mi.get("max_tokens") or 16384)}
        cost = mi.get("cost") or {}
        m["cost"] = {"input": float(cost.get("input", 0)), "output": float(cost.get("output", 0)),
                     "cacheRead": float(cost.get("cache_read", cost.get("cacheRead", 0))),
                     "cacheWrite": float(cost.get("cache_write", cost.get("cacheWrite", 0)))}
        cfg["models"] = [m]
    return cfg


def structured_mode(c: dict, api: str | None = None) -> str:
    """auto -> schema where the request format is known to take a strict JSON schema (Codex / OpenAI Responses /
    Azure Responses), prompt elsewhere. The bridge re-checks with the model's real api."""
    so = c.get("structured_output") or "auto"
    if so != "auto":
        return so
    provider, _ = split_model(c.get("model"))
    if provider in (CODEX, "openai", "azure-openai-responses") or (api or c.get("api")) == "openai-responses":
        return "schema"
    return "prompt"


def bridge_candidate(c: dict, j: dict) -> dict:
    """One entry of the bridge request's `candidates` list."""
    reasoning = c.get("reasoning")
    if reasoning is None:   # legacy key: reasoning_effort "none" == thinking off
        eff = j.get("reasoning_effort", "none")
        reasoning = "off" if eff in (None, "none") else (eff if eff in THINKING else "off")
    out = {"model": c.get("model"), "reasoning": reasoning, "structured_output": c.get("structured_output") or "auto",
           "max_tokens": int(c.get("max_tokens") or j.get("max_tokens", 500)),
           "request_timeout_ms": int(float(c.get("request_timeout_s") or 60) * 1000)}
    if c.get("reasoning") is None and split_model(c.get("model"))[0] == CODEX:
        out["reasoning_effort"] = j.get("reasoning_effort", "none")     # Codex: today's exact effort value
    if c.get("temperature") is not None:
        out["temperature"] = float(c["temperature"])
    pc = provider_config(c)
    if pc is not None:
        out["provider_config"] = pc
    return out


def meter_kind(j: dict) -> str:
    """codex | none. "auto" (the default) means codex when any candidate is an openai-codex model."""
    m = j.get("meter", "auto")
    if m != "auto":
        return m
    return "codex" if any(split_model(c.get("model"))[0] == CODEX for c in candidates(j)) else "none"


def pi_cli_argv(c: dict) -> list[str] | None:
    """The equivalent manual `pi` smoke-test command for a candidate that uses a built-in provider as is
    (custom endpoints only exist in train-snitch's in-memory registration, so they get None)."""
    if provider_config(c) is not None and not (set(provider_config(c)) <= {"apiKey"}):
        return None
    argv = ["pi", "-p", "--no-session", "-nt", "-nc", "-ne", "-ns", "-np", "--model", str(c.get("model"))]
    r = c.get("reasoning")
    if r:
        argv += ["--thinking", r]
    return argv + ["Reply with the word ok."]


def describe(j: dict) -> list[dict]:
    """Human/agent-readable summary of the resolved chain (no secrets: only env var NAMES)."""
    out = []
    for i, c in enumerate(candidates(j)):
        bc = bridge_candidate(c, j)
        pc = bc.get("provider_config") or {}
        out.append({"order": i, "model": c.get("model"), "endpoint": pc.get("baseUrl") or "Pi built-in",
                    "api": pc.get("api") or "Pi built-in",
                    "auth": ("env " + c["api_key_env"]) if c.get("api_key_env") else
                            ("none (local)" if c.get("no_auth") else "Pi login / Pi's provider env var"),
                    "reasoning": bc["reasoning"], "structured_output": structured_mode(c),
                    "pi_test_argv": pi_cli_argv(c)})
    return out
