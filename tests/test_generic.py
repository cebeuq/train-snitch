"""Generic adapter tests: log-format parsers, the read-only file reader (incl. TensorBoard event files and W&B run
dirs), local and remote (ssh) sources through a fake ssh, provider billing parsing, and the generic rules.
No network: `ssh`, `nvidia-smi`, `dmesg`, `vastai` and `runpodctl` are small fake scripts in a temp dir."""
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from train_snitch import config as cfgmod, parsers, reader, rules, timeutil  # noqa: E402
from train_snitch.adapters.generic import GenericAdapter  # noqa: E402
from train_snitch.engine import Engine  # noqa: E402
from train_snitch.probes import ProbeRunner  # noqa: E402

timeutil.set_tz("America/Los_Angeles")


# ---- a tiny TFRecord / Event writer (the format TensorBoard and torch.utils.tensorboard write) -------------
def _vint(n):
    out = b""
    while True:
        b = n & 0x7F
        n >>= 7
        out += bytes([b | (0x80 if n else 0)])
        if not n:
            return out


def _fld(fn, wt, payload):
    key = _vint(fn << 3 | wt)
    if wt == 2:
        return key + _vint(len(payload)) + payload
    return key + payload


def tf_event(step, tag, value, tensor=False):
    if tensor:   # TF2 style: TensorProto(dtype=DT_FLOAT, float_val=[value])
        val = _fld(1, 2, tag.encode()) + _fld(8, 2, _fld(1, 0, _vint(1)) + _fld(5, 5, struct.pack("<f", value)))
    else:        # simple_value
        val = _fld(1, 2, tag.encode()) + _fld(2, 5, struct.pack("<f", value))
    ev = _fld(1, 1, struct.pack("<d", 1793500000.0 + step)) + _fld(2, 0, _vint(step)) + _fld(5, 2, _fld(1, 2, val))
    return struct.pack("<Q", len(ev)) + b"\0" * 4 + ev + b"\0" * 4   # CRCs are not checked by the reader


FAKE_SSH = """#!/bin/sh
# fake ssh: drop options and the target, run the remote command locally with fake tools first on PATH
while [ $# -gt 1 ]; do
  case "$1" in -o|-F|-p|-i) shift 2;; *) if [ $# -gt 1 ]; then shift; fi;; esac
done
PATH="{bin}:$PATH" exec sh -c "$1"
"""
DENIED_SSH = "#!/bin/sh\necho 'root@203.0.113.7: Permission denied (publickey).' >&2\nexit 255\n"


def write(p, s, mode=0o644):
    with open(p, "w") as f:
        f.write(s)
    os.chmod(p, mode)


class ParserTests(unittest.TestCase):
    def test_hf(self):
        rows = parsers.log_rows(["{'loss': 0.5312, 'grad_norm': 1.25, 'learning_rate': 2e-05, 'epoch': 0.4}",
                                 "{'loss': nan, 'grad_norm': nan, 'learning_rate': 2e-05, 'epoch': 0.41}",
                                 "{'eval_loss': 0.61, 'eval_runtime': 12.0, 'epoch': 0.5}", "some other line"], "hf")
        self.assertEqual([r["kind"] for r in rows], ["train", "train", "val"])
        self.assertAlmostEqual(rows[0]["loss"], 0.5312)
        self.assertTrue(math.isnan(rows[1]["loss"]))
        self.assertEqual([r["step"] for r in rows], [1, 2, 3])

    def test_lightning_deepspeed_keyvalue_regex(self):
        lt = parsers.log_rows(["Epoch 2:  45%|####5     | 450/1000 [01:02<01:10,  7.2it/s, loss=0.123, v_num=0]"], "auto")
        self.assertEqual((lt[0]["step"], lt[0]["loss"]), (450, 0.123))
        ds = parsers.log_rows([" iteration      100/    2000 | consumed samples: 3200 | lm loss: 2.104E+00 | loss scale: 65536.0 |"])
        self.assertEqual((ds[0]["step"], round(ds[0]["loss"], 3)), (100, 2.104))
        kv = parsers.log_rows(["[2026-10-31 08:01:02] step 1200 | loss=0.4410 | lr 1e-5", "it: 7, val_loss: 0.9"])
        self.assertEqual([(r["step"], r["kind"]) for r in kv], [(1200, "train"), (7, "val")])
        rx = parsers.log_rows(["upd=55 L=inf"], regex=r"upd=(?P<step>\d+) L=(?P<loss>\S+)")
        self.assertTrue(math.isinf(rx[0]["loss"]))
        self.assertEqual(parsers.log_rows(["Traceback (most recent call last):", "  loss.backward()"]), [])

    def test_gpu_tools(self):
        nv = parsers.nvidia_rows(["0, 97, 650.12, 150000, 183000, 61", "1, [N/A], 0, 0, 0, 0"], 10.0)
        self.assertEqual([(r[1], r[2], r[3]) for r in nv], [(0, 97.0, 650.12)])
        rc = parsers.rocm_rows({"card0": {"GPU use (%)": "88", "Average Graphics Package Power (W)": "301.0"}}, 10.0)
        self.assertEqual((rc[0][1], rc[0][2], rc[0][3]), (0, 88.0, 301.0))


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d)

    def test_tail_csv_tfevents_wandb_stat_disk(self):
        d = self.d
        write(os.path.join(d, "a.log"), "one\ntwo\n")
        write(os.path.join(d, "m.csv"), "step,loss\n1,0.5\n2,0.4\n")
        os.makedirs(os.path.join(d, "tb"))
        with open(os.path.join(d, "tb", "events.out.tfevents.1.host"), "wb") as f:
            f.write(tf_event(10, "train/loss", 0.75) + tf_event(10, "lr", 1e-5) + tf_event(20, "train/loss", 0.5, tensor=True))
        run = os.path.join(d, "wandb", "offline-run-20261031_200000-abc", "files")
        os.makedirs(run)
        write(os.path.join(run, "wandb-summary.json"), json.dumps({"_step": 300, "train/loss": 0.42}))
        write(os.path.join(run, "output.log"), "{'loss': 0.42, 'epoch': 1.0}\n")
        req = {"ops": [{"id": "a", "op": "tail", "glob": os.path.join(d, "a.log")},
                       {"id": "c", "op": "csv", "glob": os.path.join(d, "m.csv")},
                       {"id": "tb", "op": "tfevents", "glob": os.path.join(d, "tb", "**", "events.out.tfevents.*"), "tags": ["train/loss"]},
                       {"id": "w", "op": "wandb", "path": os.path.join(d, "wandb")},
                       {"id": "s", "op": "stat", "glob": os.path.join(d, "*.log")},
                       {"id": "k", "op": "disk", "path": d}, {"id": "x", "op": "exists", "path": os.path.join(d, "nope")},
                       {"id": "bad", "op": "tail", "glob": "["}]}
        r = reader.handle(req)
        res = r["results"]
        self.assertEqual(list(res["a"]["files"].values())[0], ["one", "two"])
        self.assertEqual(list(res["c"]["files"].values())[0], {"header": "step,loss", "lines": ["1,0.5", "2,0.4"]})
        self.assertEqual([(p["step"], round(p["value"], 3)) for p in res["tb"]["points"]], [(10, 0.75), (20, 0.5)])
        self.assertEqual(res["w"]["summary"]["_step"], 300)
        self.assertEqual(res["s"]["count"], 1)
        self.assertGreater(res["k"]["free_gb"], 0)
        self.assertFalse(res["x"])
        # second call: only new bytes
        with open(os.path.join(d, "a.log"), "a") as f:
            f.write("three\npart")
        with open(os.path.join(d, "tb", "events.out.tfevents.1.host"), "ab") as f:
            f.write(tf_event(30, "train/loss", float("nan")))
        r2 = reader.handle({"ops": req["ops"][:3], "offsets": r["offsets"]})
        self.assertEqual(list(r2["results"]["a"]["files"].values())[0], ["three"])
        self.assertEqual(list(r2["results"]["c"]["files"].values())[0]["lines"], [])
        pts = r2["results"]["tb"]["points"]
        self.assertEqual(len(pts), 1)
        self.assertTrue(math.isnan(pts[0]["value"]))


class SourcesTests(unittest.TestCase):
    """GenericAdapter end to end through ProbeRunner + Engine, with a local host and a fake-ssh 'box'."""

    def setUp(self):
        self.d = d = tempfile.mkdtemp()
        self.bin = os.path.join(d, "bin")
        os.makedirs(self.bin)
        write(os.path.join(self.bin, "ssh"), FAKE_SSH.format(bin=self.bin), 0o755)
        write(os.path.join(self.bin, "ssh-denied"), DENIED_SSH, 0o755)
        write(os.path.join(self.bin, "nvidia-smi"), "#!/bin/sh\necho '0, 97, 650.1, 150000, 183000, 61'\necho '1, 2, 90.0, 1000, 183000, 40'\n", 0o755)
        write(os.path.join(self.bin, "dmesg"), "#!/bin/sh\necho '[Sat Oct 31 20:01:02 2026] NVRM: Xid (PCI:0000:9b:00): 79, pid=4501, name=python'\n", 0o755)
        write(os.path.join(self.bin, "vastai"), "#!/bin/sh\nif [ \"$2\" = user ]; then echo '{\"credit\": 245.97, \"email\": \"x@y\", \"ssh_key\": \"secret\"}';"
              " else echo '[{\"id\": 12345678, \"actual_status\": \"running\", \"dph_total\": 15.21, \"jupyter_token\": \"tok\"}, {\"id\": 1}]'; fi\n", 0o755)
        write(os.path.join(self.bin, "runpodctl"), "#!/bin/sh\nif [ \"$1\" = user ]; then echo '{\"clientBalance\": 50.5}';"
              " else echo '{\"id\": \"abc\", \"desiredStatus\": \"RUNNING\", \"runtimeStatus\": \"running\", \"costPerHr\": 2.99}'; fi\n", 0o755)
        self.run_dir = os.path.join(d, "run")
        os.makedirs(os.path.join(self.run_dir, "checkpoint-100"))
        write(os.path.join(self.run_dir, "train.log"), "{'loss': 0.5, 'grad_norm': 1.0, 'learning_rate': 2e-05, 'epoch': 0.1}\n")

    def tearDown(self):
        shutil.rmtree(self.d)

    def adapter(self, sources, **extra):
        cfg = cfgmod.load(None, {"cycle_s": 60, "adapter_config": dict({
            "hosts": {"box": {"ssh": "root@203.0.113.7", "port": 2222, "options": ["-i", "/nonexistent/key"]},
                      "locked": {"ssh": "root@203.0.113.8"}},
            "ssh_bin": os.path.join(self.bin, "ssh"), "sources": sources}, **extra)})
        ad = GenericAdapter(cfg)
        return ad, cfg

    def cycle(self, ad, cfg, eng, runner, t):
        res = runner.run_due(t)
        raw = ad.collect(res, eng.hist, runner)
        view = ad.normalize(raw, eng.hist)
        recs, active = eng.process(t, view)
        return raw, view, recs, active

    def test_remote_and_local_sources(self):
        rd = self.run_dir
        srcs = [{"id": "trainer", "type": "log", "path": os.path.join(rd, "train.log"), "host": "box", "format": "hf"},
                {"id": "ckpt", "type": "checkpoints", "glob": os.path.join(rd, "checkpoint-*"), "host": "box", "stale_min": 90},
                {"id": "disk", "type": "disk", "path": rd, "host": "box"},
                {"id": "gpu", "type": "gpu", "host": "box"},
                {"id": "kernel", "type": "kernel_log", "host": "box"},
                {"id": "vast", "type": "vast", "instance_id": 12345678, "credit": True, "vastai": os.path.join(self.bin, "vastai")},
                {"id": "runpod", "type": "runpod", "pod_id": "abc", "balance": True, "runpodctl": os.path.join(self.bin, "runpodctl")},
                {"id": "done", "type": "done_file", "path": os.path.join(rd, "DONE")}]
        ad, cfg = self.adapter(srcs)
        names = sorted(p.name for p in ad.probes())
        self.assertEqual(names, ["cmd:gpu", "cmd:kernel", "cmd:runpod", "cmd:runpod:credit", "cmd:vast", "cmd:vast:credit",
                                 "files@box", "files@local"])
        ssh_argv = next(p for p in ad.probes() if p.name == "cmd:gpu").argv
        self.assertEqual(ssh_argv[1:5], ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15"])
        self.assertIn("root@203.0.113.7", ssh_argv)
        eng = Engine(rules.enabled(cfg), cfg)
        runner = ProbeRunner(ad.probes(), {})
        t = time.time()
        raw, view, recs, active = self.cycle(ad, cfg, eng, runner, t)
        self.assertEqual(raw["source_errors"], {})
        self.assertEqual([r["loss"] for r in view["train_rows"]], [0.5])
        self.assertEqual(view["gpu_util_now"], {"0": 97.0, "1": 2.0})
        self.assertEqual(raw["credit_usd"], 50.5)                    # last provider read wins; vast also parsed
        self.assertEqual(sorted(str(i["id"]) for i in view["instances"]), ["12345678", "abc"])
        self.assertNotIn("jupyter_token", json.dumps(raw))           # allowlisted fields only
        self.assertNotIn("secret", json.dumps(raw))
        self.assertIn("checkpoint-100", raw["checkpoints"]["ckpt"]["newest"])
        self.assertGreater(view["disk_free_gb"], 0)
        self.assertIn("log:xid", [r["rule"] for r in recs])
        # next cycle: NaN appended remotely -> only the new line is read, and the NaN is caught
        with open(os.path.join(rd, "train.log"), "a") as f:
            f.write("{'loss': nan, 'grad_norm': nan, 'learning_rate': 2e-05, 'epoch': 0.11}\n")
        raw, view, recs, active = self.cycle(ad, cfg, eng, runner, t + 61)
        self.assertEqual(len(view["train_rows"]), 1)
        self.assertIn("loss_nonfinite", [r["rule"] for r in recs])
        self.assertIn("log:grad_nonfinite", [r["rule"] for r in recs])
        open(os.path.join(rd, "DONE"), "w").close()
        raw, view, recs, active = self.cycle(ad, cfg, eng, runner, t + 122)
        self.assertEqual(ad.done(view, eng.hist, t + 122), "done marker present")

    def test_gpu_ids_filter(self):
        ad, _ = self.adapter([{"id": "gpu", "type": "gpu", "host": "box", "ids": [0, "GPU-abc"]}])
        argv = next(p for p in ad.probes() if p.name == "cmd:gpu").argv
        self.assertIn("nvidia-smi -i 0,GPU-abc --query-gpu=", argv[-1])
        ad, _ = self.adapter([{"id": "gpu", "type": "gpu", "host": "box"}])
        self.assertNotIn(" -i ", next(p for p in ad.probes() if p.name == "cmd:gpu").argv[-1])

    def test_unreadable_source_and_process_gone(self):
        marker = f"sleep 3{os.getpid() % 1000}.5"
        p = subprocess.Popen(["sh", "-c", f"exec {marker}"])
        try:
            srcs = [{"id": "trainer", "type": "log", "path": os.path.join(self.run_dir, "train.log")},
                    {"id": "proc", "type": "process", "pattern": marker.replace(".", "\\.")},
                    {"id": "remote-log", "type": "log", "path": "/x/train.log", "host": "locked"}]
            ad, cfg = self.adapter(srcs)
            ad.acfg["ssh_bin"] = os.path.join(self.bin, "ssh-denied")
            eng = Engine(rules.enabled(cfg), cfg)
            runner = ProbeRunner(ad.probes(), {})
            t = time.time()
            got = []
            for i in range(4):
                if i == 2:
                    p.kill()
                    p.wait()
                _, view, recs, _ = self.cycle(ad, cfg, eng, runner, t + 60 * i)
                got.append([(r["rule"], r["severity"], r["status"]) for r in recs])
            self.assertTrue(view["processes"]["proc"]["alive"] is False)
            flat = [x for g in got for x in g]
            self.assertIn(("source_unreadable", "warn", "new"), flat)
            self.assertIn(("process_gone", "critical", "new"), got[3])
            su = [r for g in [self.cycle(ad, cfg, eng, runner, t + 300)[2]] for r in g]
            self.assertEqual(su, [])                                 # deduped: nothing new
        finally:
            if p.poll() is None:
                p.kill()

    def test_checkpoint_stale_and_gpus_idle_rules(self):
        cfg = cfgmod.load(None)
        eng = Engine(rules.enabled(cfg), cfg)
        t0 = 1_793_500_000
        out = []
        for i in range(6):
            t = t0 + 600 * i
            v = {"train_rows": [{"step": 10 + i, "loss": 0.5}],
                 "checkpoints": {"ckpt": {"newest": "c-1", "newest_mtime": t0 - 3600, "count": 1, "stale_min": 90, "now": t}},
                 "gpu_samples": [[t, 0, 0.0, 80, "gpu0 idle"], [t, 1, 1.0, 80, "gpu1 idle"]],
                 "planned_rate_usd_h": 15.21}
            recs, _ = eng.process(t, v)
            out.append([(r["rule"], r["status"]) for r in recs])
        flat = [x for o in out for x in o]
        self.assertEqual(out[:2], [[], []])                          # 60-70 min old (limit 90); GPUs idle < 20 min
        self.assertIn(("checkpoint_stale", "new"), flat)
        self.assertIn(("gpus_idle", "new"), flat)


if __name__ == "__main__":
    unittest.main()
