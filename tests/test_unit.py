"""Unit tests: engine state machine, rules, parsing, channels, judge guard, Pi bridge (offline faux provider),
probe runner, tail command, and one end-to-end watcher run on temp files. No network, no model calls."""
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from train_snitch import config as cfgmod, timeutil, rules  # noqa: E402
from train_snitch.adapters.generic import gpu_rows, read_new  # noqa: E402
from train_snitch.channels import Channels  # noqa: E402
from train_snitch.engine import Engine, Finding  # noqa: E402
from train_snitch.judge import Judge, find_pi_pkg  # noqa: E402
from train_snitch.probes import Probe, ProbeRunner  # noqa: E402

timeutil.set_tz("America/Los_Angeles")


def scripted(seq):
    """A rule whose output per cycle comes from a list (None = unreadable input)."""
    it = iter(seq)

    def r(t, v, h, cfg):
        return next(it)
    r.rule_name = "scripted"
    return r


def F(sev="warn", **kw):
    return Finding("x", sev, f"x is {sev}", ["ev"], "do y", **kw)


class EngineTests(unittest.TestCase):
    def run_seq(self, seq, step=60, cfg=None):
        eng = Engine([scripted(seq)], cfg or {})
        out = []
        for i in range(len(seq)):
            recs, _ = eng.process(1000 + i * step, {})
            out.append([(r["status"], r["severity"]) for r in recs])
        return out

    def test_new_ongoing_escalate_resolve(self):
        out = self.run_seq([[F()], [F()], [F("critical")], None, [F("critical")], []])
        self.assertEqual(out, [[("new", "warn")], [], [("escalated", "critical")], [], [], [("resolved", "info")]])

    def test_unreadable_input_holds(self):
        out = self.run_seq([[F()], None, None, [F()]])
        self.assertEqual(out, [[("new", "warn")], [], [], []])

    def test_hysteresis(self):
        out = self.run_seq([[F()], [], [F()], [], []], cfg={"engine": {"clear_cycles": 2}})
        self.assertEqual(out, [[("new", "warn")], [], [], [], [("resolved", "info")]])

    def test_reminders_are_sparse(self):
        out = self.run_seq([[F("critical")]] * 70, step=60)   # 70 min of the same critical
        flat = [x for c in out for x in c]
        self.assertEqual(flat, [("new", "critical"), ("ongoing", "critical"), ("ongoing", "critical")])

    def test_worsened(self):
        seq = [[F(metric=0.1, worsen_factor=2)], [F(metric=0.15, worsen_factor=2)], [F(metric=0.25, worsen_factor=2)],
               [F(metric=0.3, worsen_factor=2)]]
        out = self.run_seq(seq)
        self.assertEqual(out, [[("new", "warn")], [], [("worsened", "warn")], []])

    def test_event_ttl_and_milestone(self):
        ev = lambda: [Finding("e", "critical", "boom", kind="event")]  # noqa: E731
        out = self.run_seq([ev(), [], ev()], step=3000)   # gap > ttl (45 min): a later repeat is new again
        self.assertEqual(out, [[("new", "critical")], [], [("new", "critical")]])
        out = self.run_seq([ev(), ev(), [], ev()], step=300)   # repeats inside the ttl are ongoing (quiet)
        self.assertEqual(out, [[("new", "critical")], [], [], []])
        ms = lambda: [Finding("m", "info", "passed", kind="milestone")]  # noqa: E731
        self.assertEqual(self.run_seq([ms(), ms(), [], ms()]), [[("new", "info")], [], [], []])

    def test_rule_crash_is_contained(self):
        def boom(t, v, h, cfg):
            raise RuntimeError("bad")
        eng = Engine([boom], {})
        recs, _ = eng.process(1, {})
        self.assertEqual(recs[0]["rule"], "watcher_rule_error")

    def test_state_roundtrip(self):
        eng = Engine([scripted([[F()]])], {})
        eng.process(1, {"credit_usd": 5.0, "log_lines": {"a": ["Traceback x"]}})
        st = json.loads(json.dumps(eng.state()))
        eng2 = Engine([scripted([[F()]])], {}, st)
        recs, _ = eng2.process(2, {})
        self.assertEqual(recs, [])     # restored entry: not new again
        self.assertEqual(list(eng2.hist.credit), [[1, 5.0]])


class RuleTests(unittest.TestCase):
    def setUp(self):
        self.cfg = cfgmod.load(None)

    def eng(self, rule_list):
        return Engine(rule_list, self.cfg)

    def test_parse_local_ampm(self):
        timeutil.set_tz("America/Los_Angeles")      # other tests' watchers reset the global zone
        a = timeutil.parse_local("2026-10-26 12:55:50 AM")
        b = timeutil.parse_local("2026-10-26 12:55:50 AM PDT")
        self.assertEqual(a, b)
        self.assertEqual(timeutil.fmt12(a), "12:55 AM PDT")
        self.assertIn("PM", timeutil.fmt12(timeutil.parse_local("2026-10-31 07:32 PM")))

    def test_nan_rows(self):
        rows = [json.loads('{"step": 1, "loss": NaN}'), {"step": 2, "loss": 0.5}]
        e = self.eng([rules.loss_nonfinite])
        recs, _ = e.process(1, {"train_rows": rows})
        self.assertEqual(recs[0]["severity"], "critical")
        self.assertTrue(math.isnan(rows[0]["loss"]))

    def test_benign_nccl_not_alerted(self):
        e = self.eng([rules.log_patterns])
        line = "[rank0]:[W1026 ProcessGroupNCCL.cpp:4173] Warning: An unbatched P2P op (send/recv) was called"
        recs, _ = e.process(1, {"log_lines": {"trainer": [line, "NCCL WARN Call to connect returned Connection refused"]}})
        self.assertEqual([r["rule"] for r in recs], ["log:nccl_error"])

    def test_gpu_csv(self):
        s, now = gpu_rows(["2026/10/26 01:30:03.135, 0, 115514 MiB, 100 %, 880.32 W, 65, 1770 MHz", "bad"])
        self.assertEqual(s[0][1:4], [0, 100.0, 880.32])
        self.assertEqual(now, {"0": 100.0})

    def test_credit_runway_planned_rate(self):
        e = self.eng([rules.credit_runway])
        t = 1_000_000.0
        v = {"credit_usd": 10.0, "planned_rate_usd_h": 35.0, "deadlines": {"train_stop": t + 3600}}
        recs, _ = e.process(t, v)
        self.assertEqual(recs[0]["rule"], "credit_runway")

    def test_read_new_offsets(self):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "a.log")
        with open(p, "w") as f:
            f.write("1\n2\n")
        st = {}
        self.assertEqual(read_new(p, st, "a"), ["1", "2"])
        with open(p, "a") as f:
            f.write("3\npart")
        self.assertEqual(read_new(p, st, "a"), ["3"])
        with open(p, "a") as f:
            f.write("ial\n")
        self.assertEqual(read_new(p, st, "a"), ["partial"])
        shutil.rmtree(d)


class ChannelTests(unittest.TestCase):
    def test_jsonl_latest_and_quiet(self):
        d = tempfile.mkdtemp()
        cfg = cfgmod.load(None, {"channels": {"notify": {"enabled": False}}})
        ch = Channels(cfg, d)
        calls = []
        ch._notify = lambda r, force=False: calls.append(r)
        r = {"t": 1, "time": "x", "status": "new", "severity": "critical", "rule": "a", "key": "a", "what": "boom",
             "evidence": ["e1"], "action": "fix", "since": "x", "count": 1}
        ch.emit([r, dict(r, status="ongoing", severity="warn"), dict(r, severity="warn", quiet=True, what="quiet")])
        with open(os.path.join(d, "alerts.jsonl")) as f:
            lines = f.read().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertEqual(len(calls), 1)
        with open(os.path.join(d, "latest.txt")) as f:
            self.assertIn("WARN (ongoing) x: boom", f.read())        # newest non-quiet warn+ record
        shutil.rmtree(d)


class JudgeTests(unittest.TestCase):
    def mk(self, bridge, meter=None, **jc):
        cfg = cfgmod.load(None, {"judge": dict({"enabled": True, "max_turns": 3, "every_s": 60}, **jc)})
        st = {}
        j = Judge(cfg, st, meter_fn=meter or (lambda: {"credits": 1000.0}), bridge_fn=bridge)
        j.start()
        return j, st

    @staticmethod
    def rec(sev="critical", status="new", key="r"):
        return {"t": 0, "status": status, "severity": sev, "rule": key, "key": key, "what": "w", "evidence": [], "action": ""}

    @staticmethod
    def V(dec="ok", sev="info", topic="x", **kw):
        return {"turns": 1, "verdict": dict({"decision": dec, "severity": sev, "topic": topic, "assessment": "calm",
                                             "what": f"{dec} {topic}", "evidence": ["e"], "action": "a", "confidence": 0.8}, **kw)}

    def test_every_minute_cadence_and_modes(self):
        reqs = []

        def bridge(req):
            reqs.append(req)
            return dict(self.V(), turns=req["max_turns"])
        j, st = self.mk(bridge)
        self.assertEqual(j.maybe(1000, [], [], {"digest_text": "calm"}), [])
        self.assertEqual(reqs[-1]["mode"], "single")
        self.assertIn("calm", reqs[-1]["user_message"])
        self.assertEqual(j.maybe(1030, [], [], {}), [])              # not due: 60 s cadence
        self.assertEqual(len(reqs), 1)
        j.maybe(1060, [self.rec()], [], {})                           # new rule alert -> agent call with tools
        self.assertEqual(reqs[-1]["mode"], "agent")
        self.assertEqual(st["judge"]["calls"], 4)                     # 1 + 3 turns
        for i in range(2, 6):
            j.maybe(1000 + 60 * i, [], [], {})
        self.assertEqual(len(reqs), 6)
        self.assertEqual(st["judge"]["decisions"]["ok"], 6)

    def test_skip_unchanged(self):
        reqs = []
        j, _ = self.mk(lambda r: reqs.append(r) or self.V(), skip_unchanged=True, max_quiet_min=15)
        j.maybe(1000, [], [], {"changed": True})
        j.maybe(1060, [], [], {"changed": False})
        j.maybe(1120, [], [], {"changed": True})
        j.maybe(1120 + 16 * 60, [], [], {"changed": False})           # quiet too long: call anyway
        self.assertEqual(len(reqs), 3)

    def test_snitch_topic_dedupe_and_escalation(self):
        seq = iter([self.V("snitch", "warn", "idle-billing"), self.V("snitch", "warn", "idle-billing"),
                    self.V("snitch", "critical", "idle-billing"), self.V("snitch", "critical", "Idle Billing"),
                    self.V("watch", "warn", "loss-drift"), self.V("snitch", "critical", "idle-billing")])
        j, st = self.mk(lambda r: next(seq), snitch_repeat_min=30)
        got = [j.maybe(1000 + 60 * i, [], [], {}) for i in range(5)]
        self.assertEqual([[(r["status"], r["severity"]) for r in g] for g in got],
                         [[("new", "warn")], [], [("escalated", "critical")], [], []])
        self.assertEqual(got[0][0]["source"], "judge")
        self.assertEqual(got[0][0]["key"], "judge:idle-billing")
        out = j.maybe(1000 + 33 * 60, [], [], {})                      # reminder after the repeat window
        self.assertEqual([(r["status"], r["severity"]) for r in out], [("ongoing", "critical")])
        self.assertEqual(st["judge"]["last_verdict"]["decision"], "snitch")

    def test_false_alarm_note(self):
        j, _ = self.mk(lambda r: self.V("ok", "info", "nccl", assessment="likely_false_alarm"))
        out = j.maybe(1000, [self.rec("warn", key="smoke_failed")], [], {})
        self.assertEqual(out[0]["about"], ["smoke_failed"])
        self.assertEqual(out[0]["severity"], "info")

    def test_credit_cap_switches_off_once(self):
        j, st = self.mk(lambda req: dict(self.V(), headers={"x-codex-credits-balance": "849.0"}), credit_cap=150)
        out = j.maybe(5000, [], [], {})
        self.assertEqual([r["rule"] for r in out], ["judge_off"])
        self.assertIn("over the per-run cap of 150", st["judge"]["rules_only"])
        self.assertEqual(j.maybe(99999, [self.rec()], [], {}), [])

    def test_credit_cap_meter_and_restart(self):
        bal = {"v": 1000.0}
        j, st = self.mk(lambda req: self.V(), meter=lambda: {"credits": bal["v"]}, meter_every_calls=2)
        j.maybe(1000, [], [], {})
        j.maybe(1060, [], [], {})
        bal["v"] = 800.0
        out = j.maybe(1120, [], [], {})                                 # meter read at judgment 2 -> over cap
        self.assertEqual(out[0]["rule"], "judge_off")
        j2 = Judge(j.cfg, st, meter_fn=lambda: {"credits": 800.0}, bridge_fn=lambda r: self.V())
        self.assertEqual(j2.start()[0]["rule"], "judge_off")           # a restart in the same run stays capped

    def test_consecutive_errors_back_off(self):
        j, st = self.mk(lambda req: {"turns": 1, "error": "timeout"}, max_consecutive_errors=3)
        self.assertEqual(j.maybe(1000, [], [], {}), [])
        self.assertEqual(j.maybe(1060, [], [], {}), [])
        out = j.maybe(1120, [], [], {})
        self.assertEqual(out[0]["rule"], "judge_off")
        self.assertIn("3 model errors in a row", out[0]["what"])

    def test_usage_limit_switches_off(self):
        j, st = self.mk(lambda req: {"turns": 1, "error": "429 usage_limit_reached"})
        out = j.maybe(5000, [self.rec()], [], {})
        self.assertEqual(out[-1]["rule"], "judge_off")

    def test_disabled_writes_one_info(self):
        cfg = cfgmod.load(None, {"judge": {"enabled": False}})
        out = Judge(cfg, {}, meter_fn=lambda: None, bridge_fn=lambda r: 1 / 0).start()
        self.assertEqual((out[0]["rule"], out[0]["severity"]), ("judge_off", "info"))


@unittest.skipUnless(shutil.which("node") and os.path.isdir(find_pi_pkg()), "Node + Pi not installed")
class BridgeTests(unittest.TestCase):
    """The real bridge against Pi's offline faux provider (no network)."""
    V = json.dumps({"decision": "snitch", "severity": "warn", "topic": "smoke", "assessment": "likely_false_alarm", "what": "benign",
                    "evidence": ["x"], "action": "adopt", "confidence": 0.7})

    def call(self, req):
        base = {"model": "faux/faux", "system_prompt": "judge", "user_message": "hi", "timeout_ms": 20000, "pi_pkg": find_pi_pkg()}
        cp = subprocess.run(["node", os.path.join(ROOT, "judge", "pi_judge.mjs")], input=json.dumps({**base, **req}),
                            capture_output=True, text=True, timeout=60)
        return json.loads(cp.stdout.strip().splitlines()[-1])

    def test_single(self):
        r = self.call({"mode": "single", "faux": [self.V]})
        self.assertEqual((r["ok"], r["turns"], r["verdict"]["assessment"]), (True, 1, "likely_false_alarm"))

    def test_agent_tool_then_verdict(self):
        r = self.call({"mode": "agent", "max_turns": 3, "data": {"logs": {"trainer": ["ok", "Error x"]}},
                       "faux": [{"tool": "read_log", "args": {"source": "trainer", "grep": "Error"}}, self.V]})
        self.assertEqual((r["ok"], r["turns"]), (True, 2))

    def test_agent_probe_once_and_turn_cap(self):
        probes = [{"name": "job", "argv": ["echo", "job {arg}"], "arg_pattern": "^(train)$", "description": "d"}]
        r = self.call({"mode": "agent", "max_turns": 2, "probes": probes,
                       "faux": [{"tool": "run_probe", "args": {"name": "job", "arg": "train"}},
                                {"tool": "run_probe", "args": {"name": "job", "arg": "train"}}, self.V]})
        self.assertEqual(r["probes_run"], ["job train"])
        self.assertFalse(r["ok"])
        self.assertIn("turn cap", r["error"])


class ProbeTests(unittest.TestCase):
    def test_interval_timeout_parse(self):
        st = {}
        pr = ProbeRunner([Probe("j", argv=["sh", "-c", "echo noise; echo '{\"a\": 1}'; exit 1"], every_s=300, parse="json_last_line", ok_rc=(0, 1)),
                          Probe("slow", argv=["sleep", "5"], timeout_s=0.5)], st)
        res = pr.run_due(1000)
        self.assertEqual(res["j"].data, {"a": 1})
        self.assertTrue(res["j"].ok)
        self.assertFalse(res["slow"].ok)
        self.assertEqual(sorted(pr.run_due(1100)), ["slow"])     # j not due for 300 s
        pr2 = ProbeRunner([Probe("login", argv=["true"], on_demand=True)], {})
        self.assertEqual(pr2.run_due(1000), {})                 # on-demand probes never run on the schedule


class EndToEnd(unittest.TestCase):
    def test_watcher_once_and_done_and_tail(self):
        from train_snitch.watcher import Watcher
        d = tempfile.mkdtemp()
        log = os.path.join(d, "trainer.log")
        with open(log, "w") as f:
            f.write("start\n")
        tj = os.path.join(d, "train.jsonl")
        with open(tj, "w") as f:
            for i in range(20):
                f.write(json.dumps({"step": i, "loss": 0.5}) + "\n")
        cfgp = os.path.join(d, "cfg.json")
        with open(cfgp, "w") as f:
            json.dump({"name": "e2e", "out_dir": "out", "cycle_s": 1, "adapter": "generic",
                       "adapter_config": {"sources": [{"id": "trainer", "type": "log", "path": log, "format": "none"},
                                                      {"id": "metrics", "type": "jsonl", "path": tj},
                                                      {"id": "done", "type": "done_file", "path": os.path.join(d, "DONE")}]},
                       "judge": {"enabled": False}, "channels": {"notify": {"enabled": False}}}, f)
        cfg = cfgmod.load(cfgp)
        Watcher(cfg).run(once=True)
        with open(log, "a") as f:
            f.write("RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB\n")
        with open(tj, "a") as f:
            f.write('{"step": 20, "loss": NaN}\n')
        open(os.path.join(d, "DONE"), "w").close()
        reason = Watcher(cfgmod.load(cfgp)).run()
        self.assertIn("done marker", reason)
        with open(os.path.join(d, "out", "alerts.jsonl")) as f:
            recs = [json.loads(x) for x in f]
        rules_seen = [r["rule"] for r in recs]
        for want in ("watcher_start", "log:oom", "loss_nonfinite", "watcher_exit"):
            self.assertIn(want, rules_seen)
        with open(os.path.join(d, "out", "status.json")) as f:
            st = json.load(f)
        self.assertIn("exited", st)
        # tail replays from the start and exits at the exit record
        from train_snitch.__main__ import main
        buf = io.StringIO()
        with redirect_stdout(buf):
            main(["tail", os.path.join(d, "out"), "--from-start"])
        txt = buf.getvalue()
        self.assertIn("CRITICAL new", txt)
        self.assertIn("watcher exiting", txt)
        shutil.rmtree(d)

    @unittest.skipUnless(shutil.which("node"), "node not installed")
    def test_watcher_with_judge_fake_bridge(self):
        from train_snitch.watcher import Watcher
        d = tempfile.mkdtemp()
        fake = os.path.join(d, "fake_bridge.mjs")
        with open(fake, "w") as f:
            f.write("import {readFileSync} from 'node:fs'; const r = JSON.parse(readFileSync(0,'utf8'));"
                    "const v = {decision:'snitch', severity:'warn', topic:'gpu-idle', assessment:'real', what:'GPUs idle, '+r.mode,"
                    " evidence:[r.user_message.split('\\n')[0].slice(0,40)], action:'look', confidence:0.9};"
                    "console.log(JSON.stringify({ok:true, verdict:v, turns:1, usage:{input:10, output:5, cost:0}, headers:{}}));")
        log = os.path.join(d, "trainer.log")
        with open(log, "w") as f:
            f.write("step 1 loss=0.5\n")
        cfgp = os.path.join(d, "cfg.json")
        with open(cfgp, "w") as f:
            json.dump({"name": "e2e-judge", "out_dir": "out", "cycle_s": 60, "adapter": "generic",
                       "adapter_config": {"sources": [{"id": "trainer", "type": "log", "path": log}]},
                       "judge": {"enabled": True, "meter": "none", "bridge": fake},
                       "channels": {"notify": {"enabled": False}}}, f)
        Watcher(cfgmod.load(cfgp)).run(once=True)
        with open(os.path.join(d, "out", "alerts.jsonl")) as f:
            recs = [json.loads(x) for x in f]
        j = [r for r in recs if r["source"] == "judge"]
        self.assertEqual([(r["rule"], r["status"], r["severity"]) for r in j], [("judge:gpu-idle", "new", "warn")])
        self.assertIn("Run:", j[0]["evidence"][0])                     # the digest reached the bridge
        with open(os.path.join(d, "out", "status.json")) as f:
            st = json.load(f)
        self.assertEqual(st["judge_last"]["decision"], "snitch")
        with open(os.path.join(d, "out", "judge.jsonl")) as f:
            self.assertEqual(len(f.read().splitlines()), 1)
        shutil.rmtree(d)


if __name__ == "__main__":
    unittest.main()
