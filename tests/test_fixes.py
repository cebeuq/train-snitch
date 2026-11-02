"""Regression tests: watcher survives adapter/judge exceptions, tail exits when the watcher vanishes, secret
redaction, forced TEST delivery."""
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from train_snitch import config as cfgmod, timeutil  # noqa: E402
from train_snitch.channels import Channels  # noqa: E402
from train_snitch.engine import Engine  # noqa: E402
from train_snitch.judge import Judge  # noqa: E402
from train_snitch.redact import redact, deep  # noqa: E402

timeutil.set_tz("America/Los_Angeles")

SECRETS = [
    "hf_" + "a1B2c3D4e5F6g7H8i9J0kLmNoP",
    "sk-proj-" + "Abcdefghijklmnopqrstuvwx12345",
    "ghp_" + "abcdefghijklmnopqrstuvwxyz0123456789",
    "github_pat_" + "11ABCDEFG0123456789_abcdefghijklmnop",
    "AKIA" + "ABCDEFGHIJKLMNOP",
    "xoxb-" + "123456789012-abcdefghij",
]


def _generic_cfg(d, **extra):
    log = os.path.join(d, "trainer.log")
    with open(log, "w") as f:
        f.write("start\n")
    cfgp = os.path.join(d, "cfg.json")
    c = {"name": "fx", "out_dir": "out", "cycle_s": 0.05, "adapter": "generic",
         "adapter_config": {"sources": [{"id": "trainer", "type": "log", "path": log, "format": "none"},
                                        {"id": "done", "type": "done_file", "path": os.path.join(d, "DONE")}]},
         "judge": {"enabled": False}, "channels": {"notify": {"enabled": False}}}
    c.update(extra)
    with open(cfgp, "w") as f:
        json.dump(c, f)
    return cfgp


def _alerts(d):
    with open(os.path.join(d, "out", "alerts.jsonl")) as f:
        return [json.loads(x) for x in f]


class WatcherFaults(unittest.TestCase):
    def test_one_adapter_exception_is_reported_and_survived(self):
        from train_snitch.watcher import Watcher
        d = tempfile.mkdtemp()
        w = Watcher(cfgmod.load(_generic_cfg(d)))
        real, n = w.adapter.collect, {"i": 0}

        def flaky(*a, **k):
            n["i"] += 1
            if n["i"] == 1:
                raise OSError("injected: NAS mount vanished")
            if n["i"] == 3:
                open(os.path.join(d, "DONE"), "w").close()
            return real(*a, **k)
        w.adapter.collect = flaky
        reason = w.run()
        self.assertIn("done marker", reason)
        recs = _alerts(d)
        ce = [(r["status"], r["severity"]) for r in recs if r["rule"] == "watcher_cycle_error"]
        self.assertEqual(ce, [("new", "warn"), ("resolved", "info")])
        self.assertIn("injected", " ".join(recs[[r["rule"] for r in recs].index("watcher_cycle_error")]["evidence"]))
        self.assertEqual(recs[-1]["rule"], "watcher_exit")
        shutil.rmtree(d)

    def test_repeated_errors_give_up_with_exit_record(self):
        from train_snitch.watcher import Watcher
        d = tempfile.mkdtemp()
        w = Watcher(cfgmod.load(_generic_cfg(d, lifecycle={"max_cycle_errors": 3})))

        def boom(*a, **k):
            raise RuntimeError("injected forever")
        w.adapter.collect = boom
        reason = w.run()
        self.assertTrue(reason.startswith("crashed: 3 failed cycles in a row"), reason)
        recs = _alerts(d)
        self.assertEqual([r["rule"] for r in recs if r["rule"] == "watcher_cycle_error"], ["watcher_cycle_error"])  # deduped
        self.assertEqual((recs[-1]["rule"], recs[-1]["severity"]), ("watcher_exit", "warn"))
        self.assertFalse(os.path.exists(os.path.join(d, "out", "watcher.pid")))
        buf = io.StringIO()
        from train_snitch.__main__ import main
        with redirect_stdout(buf):
            main(["tail", os.path.join(d, "out"), "--from-start"])
        self.assertIn("watcher exiting: crashed", buf.getvalue())
        shutil.rmtree(d)

    def test_judge_exception_does_not_kill(self):
        from train_snitch.watcher import Watcher
        d = tempfile.mkdtemp()
        w = Watcher(cfgmod.load(_generic_cfg(d)))
        n = {"i": 0}

        def bad(*a, **k):
            n["i"] += 1
            if n["i"] >= 2:
                open(os.path.join(d, "DONE"), "w").close()
            raise ValueError("injected judge bug")
        w.judge.maybe = bad
        self.assertIn("done marker", w.run())
        self.assertIn("watcher_cycle_error", [r["rule"] for r in _alerts(d)])
        shutil.rmtree(d)


class TailGone(unittest.TestCase):
    def test_tail_exits_when_watcher_pid_is_dead(self):
        d = tempfile.mkdtemp()
        p = subprocess.Popen(["true"])
        p.wait()
        with open(os.path.join(d, "alerts.jsonl"), "w") as f:
            f.write(json.dumps({"rule": "x", "severity": "warn", "status": "new", "what": "w", "time": "t"}) + "\n")
        with open(os.path.join(d, "watcher.pid"), "w") as f:
            f.write(str(p.pid))
        with open(os.path.join(d, "status.json"), "w") as f:
            json.dump({"pid": p.pid}, f)
        from train_snitch.__main__ import main
        buf = io.StringIO()
        with redirect_stdout(buf):
            main(["tail", d])
        self.assertIn("WATCHER GONE", buf.getvalue())
        shutil.rmtree(d)


class Redaction(unittest.TestCase):
    def test_patterns(self):
        for s in SECRETS:
            self.assertNotIn(s, redact(f"x {s} y"), s)
        cases = ["Authorization: Bearer abcdef1234567890ABCDEF", "WANDB_API_KEY=0123456789abcdef0123456789abcdef01234567",
                 "url https://host/x?token=SuperSecretValue99&b=1", "password: hunter2hunter2",
                 "wandb: key 0123456789abcdef0123456789abcdef01234567",
                 "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXk\n-----END OPENSSH PRIVATE KEY-----",
                 "git clone https://user:pAssw0rd123@github.com/x/y"]
        for s in cases:
            out = redact(s)
            self.assertIn("[REDACTED]", out, s)
            for secret in ("abcdef1234567890ABCDEF", "0123456789abcdef0123456789abcdef01234567", "SuperSecretValue99",
                           "hunter2hunter2", "b3BlbnNzaC1rZXk", "pAssw0rd123"):
                self.assertNotIn(secret, out)
        benign = ["step 120 loss=0.4312 grad_norm=1.2 lr=1e-5", "tokens 2100 in / 23 out", "max_tokens=500",
                  "auth_errors_15m: 0", "instance 12345678 (vast, 2 GPUs)", "Permission denied (publickey)"]
        for s in benign:
            self.assertEqual(redact(s), s)
        self.assertEqual(deep({"a": ["x " + SECRETS[0]], "n": 1})["a"][0], "x [REDACTED]")

    def test_everywhere(self):
        d = tempfile.mkdtemp()
        cfg = cfgmod.load(None, {"channels": {"notify": {"enabled": False}}})
        eng = Engine([], cfg)
        eng.process(1000, {"log_lines": {"trainer": [f"Error: login failed with {SECRETS[0]}"]}})
        st = json.dumps(eng.state())
        self.assertNotIn(SECRETS[0], st)
        self.assertIn("[REDACTED]", st)
        ch = Channels(cfg, d)
        ch.emit([{"t": 1, "time": "x", "status": "new", "severity": "warn", "rule": "a", "key": "a",
                  "what": f"leaked {SECRETS[1]}", "evidence": [SECRETS[2]], "action": "", "since": "x", "count": 1}])
        for fn in ("alerts.jsonl", "latest.txt"):
            with open(os.path.join(d, fn)) as f:
                txt = f.read()
            self.assertNotIn(SECRETS[1], txt)
            self.assertNotIn(SECRETS[2], txt)
        seen = {}

        def bridge(req):
            seen.update(req)
            return {"verdict": {"decision": "ok", "severity": "info"}, "turns": 1}
        jcfg = cfgmod.load(None, {"judge": {"enabled": True, "meter": "none"}})
        j = Judge(jcfg, {}, bridge_fn=bridge)
        j.start()
        rec = {"t": 0, "status": "new", "severity": "warn", "rule": "r", "key": "r", "what": "w", "evidence": [], "action": ""}
        j.maybe(5000, [rec], [], {"digest_text": f"digest {SECRETS[3]}", "data": {"logs": {"t": [SECRETS[4]]}},
                                  "probes": [{"name": "p", "argv": ["echo", "{arg}"], "description": "d"}]})
        blob = json.dumps(seen)
        for s in SECRETS[3:5]:
            self.assertNotIn(s, blob)
        self.assertIn("redact.py", " ".join(seen["probes"][0]["argv"]))    # probe output is filtered too
        out = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "train_snitch", "redact.py"), "--", "echo", SECRETS[5]],
                             capture_output=True, text=True).stdout
        self.assertNotIn(SECRETS[5], out)
        shutil.rmtree(d)


class TestAlert(unittest.TestCase):
    def test_forced_delivery_through_every_channel(self):
        from train_snitch.watcher import test_record
        d = tempfile.mkdtemp()
        hook = os.path.join(d, "hook.jsonl")
        cfg = cfgmod.load(None, {"channels": {"notify": {"enabled": True},
                                              "command": {"argv": ["/usr/bin/tee", "-a", hook]}}})
        ch = Channels(cfg, d)
        notes = []
        ch._notify = lambda r, force=False: notes.append((r["rule"], force))
        ch.emit([test_record(cfg)])                 # info, not forced: file only
        self.assertEqual(notes, [])
        ch.emit([test_record(cfg)], force=True)
        self.assertEqual(notes, [("test", True)])
        with open(hook) as f:
            self.assertTrue(json.loads(f.read().splitlines()[-1])["test"])
        with open(os.path.join(d, "latest.txt")) as f:
            self.assertIn("TEST alert", f.read())
        from train_snitch.__main__ import main
        with open(os.path.join(d, "alerts.jsonl"), "a") as f:
            f.write(json.dumps({"rule": "watcher_exit", "severity": "info", "status": "new", "what": "bye", "time": "t"}) + "\n")
        buf = io.StringIO()
        with redirect_stdout(buf):
            main(["tail", d, "--from-start"])
        self.assertIn("TEST alert", buf.getvalue())          # info, but tail always shows TEST records
        shutil.rmtree(d)


class Webhook(unittest.TestCase):
    def test_webhook_posts_json(self):
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer
        got = []

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                got.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                self.send_response(204)
                self.end_headers()

            def log_message(self, *a):
                pass
        srv = HTTPServer(("127.0.0.1", 0), H)
        th = threading.Thread(target=srv.handle_request, daemon=True)
        th.start()
        from train_snitch.watcher import test_record
        d = tempfile.mkdtemp()
        cfg = cfgmod.load(None, {"channels": {"notify": {"enabled": False},
                                              "webhook": {"url": f"http://127.0.0.1:{srv.server_port}/hook"}}})
        Channels(cfg, d).emit([test_record(cfg)], force=True)
        th.join(5)
        srv.server_close()
        self.assertTrue(got and got[0]["test"])
        shutil.rmtree(d)


if __name__ == "__main__":
    unittest.main()
