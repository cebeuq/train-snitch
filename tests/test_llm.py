"""Judge model config (train_snitch/llm.py): validation, mapping onto Pi's provider config, the unchanged default
request, fallback order / cooldown / spend cap in the Python judge, and the real Pi bridge's model chain against
Pi's offline faux provider and a fake OpenAI-compatible server on 127.0.0.1. No network, no model calls."""
import glob
import http.server
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from train_snitch import config as cfgmod, llm  # noqa: E402
from train_snitch.judge import Judge, find_pi_pkg  # noqa: E402

VERDICT = {"decision": "snitch", "severity": "warn", "topic": "gpu-idle", "assessment": "real", "what": "GPUs idle",
           "evidence": ["util 0"], "action": "look", "confidence": 0.8}


def judge_block(**kw):
    return cfgmod.load(None, {"judge": dict({"enabled": True}, **kw)})["judge"]


class ValidateTests(unittest.TestCase):
    def test_defaults_are_valid_and_legacy(self):
        j = judge_block()
        self.assertEqual(llm.validate(j), ([], []))
        self.assertFalse(llm.uses_candidates(j))
        self.assertEqual(llm.meter_kind(j), "codex")

    def test_examples_all_validate(self):
        files = sorted(glob.glob(os.path.join(ROOT, "examples", "judge-configs", "*.json")))
        self.assertGreaterEqual(len(files), 15)
        for p in files:
            cfg = cfgmod.load(p)
            errs, _ = llm.validate(cfg["judge"])
            self.assertEqual(errs, [], p)
            self.assertTrue(llm.uses_candidates(cfg["judge"]) or cfg["judge"]["model"].startswith("openai-codex/"), p)

    def test_refuses_keys_commands_and_bad_values(self):
        cases = {
            "api_key": dict(api_key="sk-abc"),
            "api_key_env": dict(api_key_env="sk-live-123"),
            "!command": dict(headers={"x-k": "!cat secret"}),
            "looks like a secret": dict(headers={"Authorization": "Bearer abc"}),
            "base_url is required": dict(api="openai-completions"),
            "api must be one of": dict(api="cohere", base_url="http://127.0.0.1:1/v1"),
            "must not carry a key": dict(base_url="https://x.example.com/v1?api_key=abc"),
            "reasoning must be": dict(reasoning="extreme"),
            "temperature": dict(temperature=5),
            "provider/id": dict(model="gpt-6-luna"),
            "fallbacks[0]": dict(fallbacks=[42]),
            "unknown keys": dict(fallbacks=[{"model": "a/b", "bridge": "x.mjs"}]),
            "fallback_on": dict(fallback_on=["sometimes"]),
            "mutually exclusive": dict(no_auth=True, api_key_env="X_KEY"),
            "meter must be": dict(meter="maybe"),
        }
        for needle, kw in cases.items():
            errs, _ = llm.validate(judge_block(**kw))
            self.assertTrue(any(needle in e for e in errs), (needle, errs))

    def test_env_refs_in_headers_are_fine(self):
        errs, _ = llm.validate(judge_block(model="gw/m", api="openai-completions", base_url="https://gw.example.com/v1",
                                           api_key_env="GW_KEY", headers={"x-api-key": "$GW_KEY", "x-team": "ml"}))
        self.assertEqual(errs, [])

    def test_spend_guard_warning(self):
        _, w = llm.validate(judge_block(model="anthropic/claude-haiku-4-5"))
        self.assertTrue(any("cost_cap_usd" in x for x in w))
        _, w = llm.validate(judge_block(model="ollama/q", api="openai-completions", base_url="http://127.0.0.1:11434/v1", no_auth=True))
        self.assertEqual(w, [])


class MappingTests(unittest.TestCase):
    """judge block -> Pi provider config (models.json shape) and bridge candidate, per provider."""

    def cand(self, **kw):
        j = judge_block(**kw)
        return [llm.bridge_candidate(c, j) for c in llm.candidates(j)], j

    def test_codex_default_keeps_effort(self):
        (c,), _ = self.cand(fallbacks=None, reasoning=None)
        self.assertEqual(c, {"model": "openai-codex/gpt-6-luna", "reasoning": "off", "reasoning_effort": "none",
                             "structured_output": "auto", "max_tokens": 500, "request_timeout_ms": 60000})

    def test_builtin_provider_untouched_or_key_env(self):
        (c,), _ = self.cand(model="google/gemini-3.5-flash")
        self.assertNotIn("provider_config", c)
        self.assertNotIn("reasoning_effort", c)
        (c,), _ = self.cand(model="anthropic/claude-haiku-4-5", api_key_env="ANTHROPIC_API_KEY", reasoning="low")
        self.assertEqual(c["provider_config"], {"apiKey": "$ANTHROPIC_API_KEY"})
        self.assertEqual(c["reasoning"], "low")

    def test_openrouter_compat_passthrough(self):
        (c,), _ = self.cand(model="openrouter/google/gemini-2.5-flash", api_key_env="OPENROUTER_API_KEY",
                            compat={"openRouterRouting": {"data_collection": "deny"}})
        self.assertEqual(c["provider_config"], {"apiKey": "$OPENROUTER_API_KEY",
                                                "compat": {"openRouterRouting": {"data_collection": "deny"}}})
        self.assertEqual(llm.split_model(c["model"]), ("openrouter", "google/gemini-2.5-flash"))

    def test_local_ollama(self):
        (c,), _ = self.cand(model="ollama/qwen3:8b", api="openai-completions", base_url="http://127.0.0.1:11434/v1",
                            no_auth=True, compat={"supportsDeveloperRole": False},
                            model_info={"context_window": 32768, "max_tokens": 1024}, temperature=0)
        pc = c["provider_config"]
        self.assertEqual((pc["baseUrl"], pc["api"], pc["apiKey"], pc["compat"]),
                         ("http://127.0.0.1:11434/v1", "openai-completions", "local", {"supportsDeveloperRole": False}))
        self.assertEqual(pc["models"][0]["id"], "qwen3:8b")
        self.assertEqual((pc["models"][0]["contextWindow"], pc["models"][0]["maxTokens"]), (32768, 1024))
        self.assertEqual(c["temperature"], 0.0)

    def test_custom_cost_and_proxy_on_builtin(self):
        (c,), _ = self.cand(model="anthropic/claude-haiku-4-5", base_url="https://proxy.example.com")
        self.assertEqual(c["provider_config"], {"baseUrl": "https://proxy.example.com"})   # catalog kept
        (c,), _ = self.cand(model="gw/m", api="anthropic-messages", base_url="https://gw.example.com", api_key_env="GW",
                            model_info={"cost": {"input": 1, "output": 5}})
        self.assertEqual(c["provider_config"]["models"][0]["cost"], {"input": 1.0, "output": 5.0, "cacheRead": 0.0, "cacheWrite": 0.0})

    def test_fallbacks_inherit_limits_not_auth(self):
        cs, j = self.cand(model="anthropic/claude-haiku-4-5", api_key_env="ANTHROPIC_API_KEY", reasoning="low",
                          max_tokens=300, fallbacks=["openai-codex/gpt-5.6-luna", {"model": "groq/openai/gpt-oss-20b",
                                                                                    "reasoning": "minimal"}])
        self.assertEqual([c["model"] for c in cs], ["anthropic/claude-haiku-4-5", "openai-codex/gpt-5.6-luna",
                                                    "groq/openai/gpt-oss-20b"])
        self.assertEqual([c["reasoning"] for c in cs], ["low", "low", "minimal"])
        self.assertEqual({c["max_tokens"] for c in cs}, {300})
        self.assertNotIn("provider_config", cs[1])
        self.assertEqual(llm.meter_kind(j), "codex")          # a Codex model in the chain turns the meter on

    def test_pi_cli_argv(self):
        self.assertEqual(llm.pi_cli_argv({"model": "google/gemini-3.5-flash", "reasoning": "low"})[-5:],
                         ["--model", "google/gemini-3.5-flash", "--thinking", "low", "Reply with the word ok."])
        self.assertIsNone(llm.pi_cli_argv({"model": "ollama/q", "api": "openai-completions", "base_url": "http://127.0.0.1:1/v1"}))
        d = llm.describe(judge_block(model="ollama/q", api="openai-completions", base_url="http://127.0.0.1:1/v1", no_auth=True))
        self.assertEqual((d[0]["auth"], d[0]["endpoint"]), ("none (local)", "http://127.0.0.1:1/v1"))


class JudgeChainTests(unittest.TestCase):
    """The Python judge with a fake bridge: unchanged default request, fallback order, cooldown, caps."""

    def mk(self, bridge_fn, **jc):
        cfg = cfgmod.load(None, {"judge": dict({"enabled": True, "every_s": 60}, **jc)})
        st = {}
        j = Judge(cfg, st, meter_fn=lambda: None, bridge_fn=bridge_fn)
        j.start()
        return j, st

    def test_default_request_is_unchanged(self):
        reqs = []
        j, _ = self.mk(lambda r: reqs.append(r) or {"turns": 1, "verdict": dict(VERDICT, decision="ok", severity="info")})
        j._pi = "/pi"
        j.maybe(1000, [], [], {"digest_text": "calm"})
        self.assertEqual(sorted(reqs[0]), sorted(["model", "mode", "max_turns", "reasoning_effort", "timeout_ms",
                                                  "system_prompt", "user_message", "data", "probes", "pi_pkg", "max_tokens"]))
        self.assertEqual((reqs[0]["model"], reqs[0]["reasoning_effort"], reqs[0]["max_tokens"], reqs[0]["timeout_ms"]),
                         ("openai-codex/gpt-6-luna", "none", 500, 90000))

    def test_chain_request_and_model_used(self):
        reqs = []

        def bridge(r):
            reqs.append(r)
            return {"turns": 1, "verdict": VERDICT, "model_used": "anthropic/claude-haiku-4-5",
                    "attempts": [{"model": "openai-codex/gpt-6-luna", "ok": False, "error_kind": "not_found"},
                                 {"model": "anthropic/claude-haiku-4-5", "ok": True}]}
        j, st = self.mk(bridge, fallbacks=["anthropic/claude-haiku-4-5"])
        out = j.maybe(1000, [], [], {})
        self.assertEqual([c["model"] for c in reqs[0]["candidates"]], ["openai-codex/gpt-6-luna", "anthropic/claude-haiku-4-5"])
        self.assertEqual(reqs[0]["fallback_on"], llm.DEFAULT_FALLBACK_ON)
        self.assertEqual(out[0]["judge_model"], "anthropic/claude-haiku-4-5")
        j.maybe(1060, [], [], {})                       # the missing model cools down: only the fallback is sent
        self.assertEqual([c["model"] for c in reqs[1]["candidates"]], ["anthropic/claude-haiku-4-5"])
        j.maybe(1000 + 16 * 60, [], [], {})             # cooldown over: primary first again
        self.assertEqual(reqs[2]["candidates"][0]["model"], "openai-codex/gpt-6-luna")

    def test_all_cooling_down_still_sends_all(self):
        reqs = []
        rl = {"turns": 1, "error": "429 rate limit", "attempts": [{"model": "a/x", "error_kind": "rate_limit"},
                                                                  {"model": "b/y", "error_kind": "no_auth"}]}
        j, st = self.mk(lambda r: reqs.append(r) or rl, model="a/x", fallbacks=["b/y"], max_consecutive_errors=5)
        j.maybe(1000, [], [], {})
        self.assertIsNone(st["judge"]["rules_only"])    # not every candidate was rate-limited: keep trying
        j.maybe(1060, [], [], {})
        self.assertEqual([c["model"] for c in reqs[1]["candidates"]], ["a/x", "b/y"])

    def test_every_codex_candidate_rate_limited_goes_rules_only(self):
        res = {"turns": 1, "error": "You have hit your ChatGPT usage limit (plus plan). Try again in ~42 min.",
               "attempts": [{"model": "openai-codex/gpt-6-luna", "error_kind": "rate_limit"},
                            {"model": "openai-codex/gpt-5.6-luna", "error_kind": "rate_limit"}]}
        j, st = self.mk(lambda r: res, fallbacks=["openai-codex/gpt-5.6-luna"])
        out = j.maybe(1000, [], [], {})
        self.assertEqual(out[-1]["rule"], "judge_off")

    def test_other_providers_rate_limited_pause_not_off(self):
        calls = []
        rl = lambda m: {"turns": 1, "error": "429", "attempts": [{"model": m, "error_kind": "rate_limit"}]}
        ok = {"turns": 1, "verdict": dict(VERDICT, decision="ok", severity="info"), "model_used": "b/y",
              "attempts": [{"model": "a/x", "error_kind": "rate_limit"}, {"model": "b/y", "ok": True}]}
        script = [ok, rl("b/y"), rl("a/x")]
        j, st = self.mk(lambda r: calls.append(r) or script.pop(0), model="a/x", fallbacks=["b/y"],
                        max_consecutive_errors=1)
        j.maybe(1000, [], [], {})                                   # a/x 429 -> cooldown; b/y answers
        self.assertEqual(j.maybe(1060, [], [], {}), [])             # only b/y sent; it 429s too
        self.assertEqual([c["model"] for c in calls[1]["candidates"]], ["b/y"])
        self.assertIsNone(st["judge"]["rules_only"])                # not switched off, not counted as an error
        self.assertEqual(st["judge"]["errors_in_row"], 0)
        self.assertEqual(j.maybe(1120, [], [], {}), [])             # paused: no call while both cool down
        self.assertEqual(len(calls), 2)
        j.maybe(1000 + 15 * 60 + 1, [], [], {})                     # a/x's cooldown is over: judging again
        self.assertEqual(len(calls), 3)
        self.assertIsNone(st["judge"]["rules_only"])

    def test_single_non_codex_model_429_does_not_switch_off(self):
        res = {"turns": 1, "error": "429 rate limit", "attempts": [{"model": "groq/m", "error_kind": "rate_limit"}]}
        j, st = self.mk(lambda r: res, model="groq/m")
        self.assertEqual(j.maybe(1000, [], [], {}), [])
        self.assertIsNone(st["judge"]["rules_only"])
        self.assertGreater(st["judge"]["paused_until"], 1000)

    def test_cost_cap_usd(self):
        res = {"turns": 1, "verdict": dict(VERDICT, decision="ok", severity="info"), "usage": {"input": 1, "output": 1, "cost": 0.6}}
        j, st = self.mk(lambda r: res, model="anthropic/claude-haiku-4-5", cost_cap_usd=1)
        self.assertEqual(j.maybe(1000, [], [], {}), [])
        out = j.maybe(1060, [], [], {})
        self.assertEqual(out[0]["rule"], "judge_off")
        self.assertIn("over the per-run cap of $1", st["judge"]["rules_only"])

    def test_pi_agent_dir_env(self):
        seen = {}
        j, _ = self.mk(None, pi_agent_dir="/tmp/pi-agent-test", bridge=sys.executable, node=sys.executable)
        real_run = subprocess.run

        def fake_run(argv, **kw):
            seen.update(kw.get("env") or {})
            return subprocess.CompletedProcess(argv, 0, stdout='{"turns":1,"error":"x"}', stderr="")
        subprocess.run = fake_run
        try:
            j._bridge({"timeout_ms": 1000})
        finally:
            subprocess.run = real_run
        self.assertEqual(seen.get("PI_CODING_AGENT_DIR"), "/tmp/pi-agent-test")

    def test_watcher_refuses_invalid_judge(self):
        from train_snitch.watcher import Watcher
        d = tempfile.mkdtemp()
        cfg = cfgmod.load(None, {"out_dir": os.path.join(d, "out"), "judge": {"enabled": True, "api_key": "sk-nope"}})
        with self.assertRaises(SystemExit) as cm:
            Watcher(cfg)
        self.assertIn("api_key is not allowed", str(cm.exception))
        self.assertFalse(os.path.exists(os.path.join(d, "out")))      # refused before touching the out_dir
        shutil.rmtree(d)


# ---------------------------------------------------------------- the real bridge
class FakeOpenAI(http.server.BaseHTTPRequestHandler):
    """Minimal OpenAI Chat Completions server (SSE). Model "busy" answers 429."""
    seen: list = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)) or b"{}")
        FakeOpenAI.seen.append({"path": self.path, "auth": self.headers.get("authorization"), "body": body})
        if body.get("model") == "busy":
            data = b'{"error":{"message":"Rate limit reached","type":"rate_limit_exceeded"}}'
            self.send_response(429)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        base = {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": body.get("model")}
        for ch in ({"index": 0, "delta": {"role": "assistant", "content": json.dumps(VERDICT)}, "finish_reason": None},
                   {"index": 0, "delta": {}, "finish_reason": "stop"}):
            self.wfile.write(b"data: " + json.dumps(dict(base, choices=[ch])).encode() + b"\n\n")
        self.wfile.write(b"data: " + json.dumps(dict(base, choices=[], usage={"prompt_tokens": 50, "completion_tokens": 20,
                                                                               "total_tokens": 70})).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


@unittest.skipUnless(shutil.which("node") and os.path.isdir(find_pi_pkg()), "Node + Pi not installed")
class BridgeChainTests(unittest.TestCase):
    FAUX = json.dumps(VERDICT)

    @classmethod
    def setUpClass(cls):
        cls.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeOpenAI)
        cls.url = f"http://127.0.0.1:{cls.srv.server_address[1]}/v1"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def call(self, req, env=None):
        base = {"model": "x/y", "mode": "single", "system_prompt": "judge", "user_message": "hi", "timeout_ms": 30000,
                "pi_pkg": find_pi_pkg(), "max_tokens": 200}
        cp = subprocess.run(["node", os.path.join(ROOT, "judge", "pi_judge.mjs")], input=json.dumps({**base, **req}),
                            capture_output=True, text=True, timeout=90, env=dict(os.environ, **(env or {})))
        return json.loads(cp.stdout.strip().splitlines()[-1])

    def local(self, model, **kw):
        j = judge_block(model=model, api="openai-completions", base_url=self.url, **kw)
        return llm.bridge_candidate(llm.candidates(j)[0], j)

    def test_fallback_order_not_found_no_auth_then_faux(self):
        cands = [{"model": "nope/missing-model", "reasoning": "off", "max_tokens": 100},
                 self.local("keyed/m1", api_key_env="TS_TEST_UNSET_KEY_VAR"),
                 {"model": "faux-judge/faux-judge", "reasoning": "off", "max_tokens": 100}]
        r = self.call({"candidates": cands, "faux": [self.FAUX]})
        self.assertEqual([a.get("error_kind") for a in r["attempts"]], ["not_found", "no_auth", None])
        self.assertEqual((r["ok"], r["model_used"], r["turns"]), (True, "faux-judge/faux-judge", 1))

    def test_fallback_on_limits_the_chain(self):
        r = self.call({"candidates": [{"model": "nope/missing-model"}, {"model": "faux-judge/faux-judge"}],
                       "faux": [self.FAUX], "fallback_on": ["rate_limit"]})
        self.assertFalse(r["ok"])
        self.assertEqual(len(r["attempts"]), 1)
        self.assertIn("model not found", r["error"])

    def test_local_openai_compatible_server(self):
        FakeOpenAI.seen.clear()
        c = self.local("ollama/qwen3:8b", no_auth=True, structured_output="schema", temperature=0,
                       compat={"supportsDeveloperRole": False, "supportsReasoningEffort": False})
        r = self.call({"candidates": [c]})
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["verdict"]["topic"], "gpu-idle")
        self.assertEqual(r["model_used"], "ollama/qwen3:8b")
        body = FakeOpenAI.seen[-1]["body"]
        self.assertEqual(body["model"], "qwen3:8b")
        self.assertEqual(body["response_format"]["type"], "json_schema")
        self.assertEqual(body["response_format"]["json_schema"]["schema"]["required"][0], "decision")
        self.assertEqual(body.get("temperature"), 0)
        self.assertNotIn("reasoning_effort", body)
        self.assertEqual(FakeOpenAI.seen[-1]["auth"], "Bearer local")
        self.assertEqual(body["messages"][0]["role"], "system")
        self.assertEqual(r["usage"]["input"], 50)

    def test_api_key_env_and_rate_limit_fallback(self):
        FakeOpenAI.seen.clear()
        busy = self.local("gw/busy", api_key_env="TS_TEST_GW_KEY")
        ok = self.local("gw2/m2", api_key_env="TS_TEST_GW_KEY", structured_output="prompt")
        r = self.call({"candidates": [busy, ok]}, env={"TS_TEST_GW_KEY": "test-key-not-secret"})
        self.assertTrue(r["ok"], r)
        self.assertEqual([a.get("error_kind") for a in r["attempts"]], ["rate_limit", None])
        self.assertEqual(r["model_used"], "gw2/m2")
        self.assertEqual({s["auth"] for s in FakeOpenAI.seen}, {"Bearer test-key-not-secret"})
        self.assertNotIn("response_format", FakeOpenAI.seen[-1]["body"])          # prompt mode: no schema
        self.assertNotIn("test-key-not-secret", json.dumps(r))                       # never echoed back

    def test_agent_mode_through_chain(self):
        r = self.call({"mode": "agent", "max_turns": 3, "data": {"logs": {"trainer": ["ok", "Error x"]}},
                       "candidates": [{"model": "nope/missing-model"}, {"model": "faux-judge/faux-judge", "reasoning": "off"}],
                       "faux": [{"tool": "read_log", "args": {"source": "trainer", "grep": "Error"}}, self.FAUX]})
        self.assertEqual((r["ok"], r["turns"], r["model_used"]), (True, 2, "faux-judge/faux-judge"))

    def test_agent_mode_applies_max_tokens_and_temperature(self):
        FakeOpenAI.seen.clear()
        c = self.local("ollama/qwen3:8b", no_auth=True, temperature=0.2, max_tokens=123,
                       compat={"supportsDeveloperRole": False, "supportsReasoningEffort": False})
        r = self.call({"mode": "agent", "max_turns": 3, "data": {"logs": {"trainer": ["ok"]}}, "candidates": [c]})
        self.assertTrue(r["ok"], r)
        body = FakeOpenAI.seen[-1]["body"]
        self.assertEqual(body.get("max_completion_tokens", body.get("max_tokens")), 123)
        self.assertEqual(body.get("temperature"), 0.2)

    def test_legacy_request_still_works(self):
        r = self.call({"faux": [self.FAUX]})               # no candidates: the original path
        self.assertEqual((r["ok"], r["turns"]), (True, 1))
        self.assertNotIn("attempts", r)


if __name__ == "__main__":
    unittest.main()
