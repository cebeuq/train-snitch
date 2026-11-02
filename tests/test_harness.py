"""Harness control: the dependency-free control layer (start / stop / list / status / alerts, safety boundary),
the MCP server through the official SDK's in-process client and over real stdio (skipped without the [mcp]
extra), and the per-harness integration files (skills identical, configs parse, tool names match). Offline."""
import asyncio
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from train_snitch import control  # noqa: E402

try:
    import mcp  # noqa: F401
    HAVE_MCP = True
except ImportError:
    HAVE_MCP = False

TOOLS = {"list_watches", "status", "latest", "tail_alerts", "check_config", "start_watch", "stop_watch", "meter"}


class Sandbox(unittest.TestCase):
    """A temp folder with configs/ and runs/, allowlisted through the env vars; one tiny rules-only run."""

    def setUp(self):
        self.d = os.path.realpath(tempfile.mkdtemp())
        self.cfgs, self.runs = os.path.join(self.d, "configs"), os.path.join(self.d, "runs")
        os.makedirs(self.cfgs)
        os.makedirs(self.runs)
        self.log = os.path.join(self.d, "train.log")
        with open(self.log, "w") as f:
            f.write("step 1 loss=0.5\nstep 2 loss=0.4\n")
        self.cfg = self.write("run.json", {"name": "demo-run", "out_dir": "../runs/demo", "cycle_s": 1,
                                           "lifecycle": {"max_hours": 0.05},
                                           "adapter_config": {"sources": [{"id": "trainer", "type": "log", "path": self.log}]},
                                           "channels": {"notify": {"enabled": False}}})
        self.env = {k: os.environ.get(k) for k in (control.ENV_CONFIG_DIRS, control.ENV_OUT_DIRS, control.ENV_FILE)}
        os.environ[control.ENV_CONFIG_DIRS] = self.cfgs
        os.environ[control.ENV_OUT_DIRS] = self.runs
        os.environ.pop(control.ENV_FILE, None)

    def tearDown(self):
        for k, v in self.env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        for pid, p in list(control._CHILDREN.items()):     # never leave a watcher behind
            if p.poll() is None:
                p.terminate()
                p.wait(timeout=20)
        control._CHILDREN.clear()
        shutil.rmtree(self.d, ignore_errors=True)

    def write(self, name, obj):
        p = os.path.join(self.cfgs, name)
        with open(p, "w") as f:
            json.dump(obj, f)
        return p

    def wait_for(self, fn, secs=20):
        t = time.time() + secs
        while time.time() < t:
            v = fn()
            if v:
                return v
            time.sleep(0.2)
        return fn()


class ControlTests(Sandbox):
    def test_start_list_status_alerts_stop(self):
        r = control.start_watch(self.cfg, rules_only=True, dry=True)
        self.assertTrue(r["started"], r)
        out = os.path.join(self.runs, "demo")
        self.assertEqual(r["out_dir"], out)
        self.assertTrue(self.wait_for(lambda: os.path.exists(os.path.join(out, "status.json"))))
        ws = control.list_watches()["watches"]
        self.assertEqual([(w["run"], w["alive"]) for w in ws], [("demo-run", True)])
        self.assertTrue(control.status("demo-run")["alive"])
        self.assertIn("cycle", control.status(out)["status"])
        self.assertIn("latest", control.latest("demo"))
        a = control.tail_alerts("demo-run", n=10)
        self.assertIn("watcher_start", [x["rule"] for x in a["alerts"]])
        with self.assertRaises(control.ControlError):              # one watcher per out_dir
            control.start_watch(self.cfg, rules_only=True, dry=True)
        s = control.stop_watch("demo-run")
        self.assertTrue(s["stopped"], s)
        self.assertIn("stopped by signal", s["exited"])
        a2 = control.tail_alerts(out, since=a["next_since"])
        self.assertEqual(a2["alerts"][-1]["rule"], "watcher_exit")
        self.assertFalse(a2["alive"])
        self.assertEqual(control.stop_watch(out)["stopped"], False)  # already gone: nothing to signal

    def test_once_with_overrides_and_test_alert(self):
        r = control.start_watch(self.cfg, overrides={"name": "renamed", "judge": {"enabled": False}}, dry=True,
                                once=True, test_alert=True)
        out = r["out_dir"]
        self.assertTrue(self.wait_for(lambda: (control._read_json(os.path.join(out, "status.json")) or {}).get("exited")))
        st = control.status(out)
        self.assertEqual((st["run"], st["alive"]), ("renamed", False))
        recs = control.tail_alerts(out, n=50, min_severity="critical")["alerts"]
        self.assertIn("test", [x["rule"] for x in recs])           # TEST records always pass the filter

    def test_boundary(self):
        outside = os.path.join(self.d, "elsewhere.json")
        with open(outside, "w") as f:
            json.dump({"name": "x"}, f)
        cases = [
            (lambda: control.start_watch(outside), "outside the allowed config folders"),
            (lambda: control.start_watch(self.write("o.json", {"out_dir": os.path.join(self.d, "x")})), "outside the allowed output"),
            (lambda: control.start_watch(self.cfg, overrides={"adapter_config": {}}), "not allowed inline"),
            (lambda: control.start_watch(self.cfg, overrides={"channels": {"command": {"argv": ["sh"]}}}), "not allowed inline"),
            (lambda: control.start_watch(self.cfg, overrides={"judge": {"node": "/bin/sh"}}), "not allowed inline"),
            (lambda: control.start_watch(self.cfg, overrides={"judge": {"base_url": "https://x.example.com/v1"}}), "not allowed inline"),
            (lambda: control.start_watch(self.cfg, overrides={"judge": {"fallbacks": [{"model": "a/b"}]}}), "only \"provider/id\" strings"),
            (lambda: control.start_watch(self.write("bad.json", {"out_dir": "../runs/bad", "judge": {"enabled": True, "api_key": "sk-x"}})), "invalid judge config"),
            (lambda: control.start_watch(os.path.join(self.cfgs, "run.yaml")), ".json"),
            (lambda: control.status(self.d), "outside the allowed output folders"),
            (lambda: control.status("no-such-run"), "no watcher named"),
            (lambda: control.tail_alerts(self.runs, min_severity="loud"), "min_severity"),
        ]
        for fn, needle in cases:
            with self.assertRaises(control.ControlError) as cm:
                fn()
            self.assertIn(needle, str(cm.exception))

    def test_stop_refuses_a_non_watcher_pid(self):
        out = os.path.join(self.runs, "fake")
        os.makedirs(out)
        p = subprocess.Popen(["sleep", "30"])
        try:
            with open(os.path.join(out, "watcher.pid"), "w") as f:
                f.write(str(p.pid))
            with self.assertRaises(control.ControlError) as cm:
                control.stop_watch(out)
            self.assertIn("not a train-snitch watcher", str(cm.exception))
            self.assertIsNone(p.poll())                                # untouched
        finally:
            p.kill()
            p.wait()

    def test_default_roots_and_home_guard(self):
        os.environ.pop(control.ENV_CONFIG_DIRS)
        os.environ.pop(control.ENV_OUT_DIRS)
        cwd = os.getcwd()
        try:
            os.chdir(self.cfgs)
            self.assertEqual(control.config_roots(), [self.cfgs])
            self.assertEqual(control.out_roots(), [self.cfgs])
            os.chdir(os.path.expanduser("~"))
            self.assertEqual(control.config_roots(), [])
            with self.assertRaises(control.ControlError) as cm:
                control.list_watches()
            self.assertIn("TRAIN_SNITCH_OUT_DIRS", str(cm.exception))
        finally:
            os.chdir(cwd)

    def test_env_file(self):
        ef = os.path.join(self.d, "keys.env")
        with open(ef, "w") as f:
            f.write("# judge keys\nexport TS_TEST_A='alpha-value'\nTS_TEST_B=beta\nnot a line\n")
        os.chmod(ef, 0o644)
        os.environ[control.ENV_FILE] = ef
        with self.assertRaises(control.ControlError) as cm:
            control.child_env()
        self.assertIn("chmod 600", str(cm.exception))
        self.assertNotIn("alpha", str(cm.exception))
        os.chmod(ef, 0o600)
        env = control.child_env()
        self.assertEqual((env["TS_TEST_A"], env["TS_TEST_B"]), ("alpha-value", "beta"))
        self.assertIn("HOME", env)

    def test_check_config(self):
        r = control.check_config(self.cfg, {"judge": {"model": "anthropic/claude-haiku-4-5", "cost_cap_usd": 1}})
        self.assertTrue(r["ok"])
        self.assertEqual(r["judge"][0]["model"], "anthropic/claude-haiku-4-5")
        self.assertEqual(r["meter"], "none")


def run_async(coro):
    return asyncio.run(coro)


@unittest.skipUnless(HAVE_MCP, "the [mcp] extra (official MCP Python SDK) is not installed")
class MCPServerTests(Sandbox):
    def test_tools_in_process(self):
        from mcp import Client
        from train_snitch.mcp_server import build_server

        async def go():
            async with Client(build_server()) as c:
                tools = await c.list_tools()
                names = {t.name for t in tools.tools}
                ro = {t.name: t.annotations.read_only_hint for t in tools.tools}
                lw = await c.call_tool("list_watches", {})
                bad = await c.call_tool("start_watch", {"config": "/etc/passwd.json"})
                ok = await c.call_tool("check_config", {"config": self.cfg})
                return names, ro, lw, bad, ok
        names, ro, lw, bad, ok = run_async(go())
        self.assertEqual(names, TOOLS)
        self.assertTrue(ro["list_watches"] and ro["tail_alerts"] and not ro["start_watch"] and not ro["stop_watch"])
        self.assertFalse(lw.is_error)
        self.assertEqual(lw.structured_content["watches"], [])
        self.assertTrue(bad.is_error)
        self.assertIn("outside the allowed config folders", bad.content[0].text)
        self.assertTrue(ok.structured_content["ok"])

    def test_start_tail_stop_over_stdio(self):
        """The real console entry over stdio, as a harness launches it."""
        from mcp import Client, StdioServerParameters
        params = StdioServerParameters(command=sys.executable, args=["-I", "-c", control.BOOT, ROOT, "mcp"],
                                       env=dict(os.environ), cwd=self.d)

        async def go():
            async with Client(params) as c:
                st = await c.call_tool("start_watch", {"config": self.cfg, "rules_only": True, "dry": True})
                out = st.structured_content["out_dir"]
                for _ in range(100):
                    s = await c.call_tool("status", {"run": "demo-run"})
                    if not s.is_error and s.structured_content.get("status"):
                        break
                    await asyncio.sleep(0.2)
                al = await c.call_tool("tail_alerts", {"run": out, "n": 5})
                sp = await c.call_tool("stop_watch", {"run": "demo-run"})
                return st, al, sp
        st, al, sp = run_async(go())
        self.assertTrue(st.structured_content["started"], st)
        self.assertIn("watcher_start", [a["rule"] for a in al.structured_content["alerts"]])
        self.assertTrue(sp.structured_content["stopped"], sp)


class IntegrationFileTests(unittest.TestCase):
    """integrations/: one canonical skill, byte-identical copies, parseable configs, matching tool names."""
    INT = os.path.join(ROOT, "integrations")

    def test_skill_copies_identical(self):
        canon = os.path.join(self.INT, "skill", "train-snitch", "SKILL.md")
        with open(canon, "rb") as f:
            want = f.read()
        copies = [p for p in glob.glob(os.path.join(self.INT, "**", "SKILL.md"), recursive=True) if p != canon]
        self.assertGreaterEqual(len(copies), 6)
        for p in copies:
            self.assertEqual(os.path.basename(os.path.dirname(p)), "train-snitch", p)
            with open(p, "rb") as f:
                self.assertEqual(f.read(), want, f"{p} differs from the canonical skill; run integrations/sync-skill.py")
        fm = re.match(r"^---\nname: (.+)\ndescription: (.+)\n---\n", want.decode())
        self.assertIsNotNone(fm)
        self.assertEqual(fm.group(1), "train-snitch")
        self.assertLessEqual(len(fm.group(2)), 1024)

    def test_skill_and_commands_name_real_tools(self):
        texts = []
        for p in glob.glob(os.path.join(self.INT, "**", "*"), recursive=True):
            if os.path.isfile(p) and p.endswith((".md", ".toml", ".mdc")):
                with open(p) as f:
                    texts.append(f.read())
        used = set(re.findall(r"\b(list_watches|status|latest|tail_alerts|check_config|start_watch|stop_watch|meter|"
                              r"snitch_\w+|watch_\w+)\(", "\n".join(texts)))
        self.assertTrue(used, "no tool names found")
        self.assertLessEqual(used, TOOLS, used - TOOLS)

    def test_configs_parse_and_register_the_server(self):
        files = [p for p in glob.glob(os.path.join(self.INT, "**", "*.json"), recursive=True)]
        files += glob.glob(os.path.join(ROOT, ".claude-plugin", "*.json"))
        files += [os.path.join(self.INT, "claude-code", ".mcp.json"),
                  os.path.join(self.INT, "claude-code", ".claude-plugin", "plugin.json")]
        self.assertGreaterEqual(len(files), 7)
        servers = 0
        for p in files:
            with open(p) as f:
                obj = json.load(f)
            blob = json.dumps(obj)
            self.assertNotRegex(blob, r"(?:sk-|hf_|ghp_)[A-Za-z0-9]{8,}", p)
            for sect in ("mcpServers", "mcp"):
                if isinstance(obj.get(sect), dict) and "train-snitch" in obj[sect]:
                    servers += 1
                    self.assertIn("train-snitch-mcp", json.dumps(obj[sect]["train-snitch"]))
        self.assertGreaterEqual(servers, 6)
        if sys.version_info >= (3, 11):
            import tomllib
            for p in glob.glob(os.path.join(self.INT, "**", "*.toml"), recursive=True):
                with open(p, "rb") as f:
                    tomllib.load(f)
            with open(os.path.join(self.INT, "codex", "config.toml"), "rb") as f:
                codex = tomllib.load(f)
            self.assertEqual(codex["mcp_servers"]["train-snitch"]["command"], "train-snitch-mcp")
            self.assertEqual(set(codex["mcp_servers"]["train-snitch"]["enabled_tools"]), TOOLS)


if __name__ == "__main__":
    unittest.main()
