"""Eval judge tests: naming / set detection, stop policy, verdict normalization, sheets, the runner end to end with a
scripted judge (alerts, judge log, credit cap, act-once). No model calls."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from train_snitch import evaljudge as ej  # noqa: E402
from train_snitch import sheet  # noqa: E402

HAVE_FFMPEG = os.path.exists(sheet.FFMPEG)
POL = dict(ej.DEFAULTS["policy"])


def V(step, s="same", l="same", conf=0.8, persist="n/a", regs=None, ss=4, ls=4, ref="better", prev=None):
    return {"step": step, "structure_score": ss, "look_score": ls, "vs_previous": {"structure": s, "look": l},
            "vs_reference": {"structure": ref, "look": ref}, "previous_regressions_persist": persist,
            "regressions": regs or [], "notable": [], "recommend": "continue", "confidence": conf, "reason": "r",
            "prev_step": prev}


def rd(p):
    with open(p) as f:
        return f.read()


def make_jpg(path, color="red", size=64):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    subprocess.run([sheet.FFMPEG, "-v", "error", "-nostdin", "-y", "-f", "lavfi", "-i", f"color=c={color}:s={size}x{size}",
                    "-frames:v", "1", path], check=True, capture_output=True)


class TestNaming(unittest.TestCase):
    def test_aitk_idx_grouping(self):
        pat = ej.DEFAULTS["source"]["pattern"]
        self.assertEqual(ej.parse_name("1793019187897__000002000_9.jpg", pat, 3), (2000, 3, 0))
        self.assertEqual(ej.parse_name("1793019187897__000000500_59.jpg", pat, 3), (500, 19, 2))
        self.assertIsNone(ej.parse_name("1793019187897__000000500_59.jpg.part", pat, 3))
        self.assertIsNone(ej.parse_name("base-03.jpg", pat, 3))

    def test_dir_layout(self):
        pat = r"^(?P<step>\d+)/p(?P<prompt>\d+)\.jpg$"
        self.assertEqual(ej.parse_name("1000/p044.jpg", pat), (1000, 44, 0))
        self.assertIsNone(ej.parse_name("1000/p044.json", pat))

    def test_sets_ready(self):
        src = dict(ej.DEFAULTS["source"], group_size=3, expect=6, partial_after_s=600, min_age_s=20)
        now = time.time()
        files = [(f"1__{500:09d}_{i}.jpg", 10, now - 100) for i in range(6)]
        files += [(f"1__{1000:09d}_{i}.jpg", 10, now - 100) for i in range(4)]          # incomplete, recent
        files += [(f"1__{1500:09d}_{i}.jpg", 10, now - 5) for i in range(6)]            # complete but too fresh
        files += [(f"1__{2000:09d}_{i}.jpg", 10, now - 700) for i in range(2)]          # incomplete, old, but last
        files += [(f"1__{2500:09d}_{i}.jpg", 0, now - 700) for i in range(6)]           # empty files ignored
        sets = ej.group_sets(files, src)
        self.assertEqual(len(sets[500]["files"]), 6)
        self.assertEqual(ej.ready_steps(sets, src, now, set()), [500])                  # 2000: no later step yet
        files += [(f"1__{3000:09d}_{i}.jpg", 10, now - 100) for i in range(6)]
        sets = ej.group_sets(files, src)
        self.assertEqual(ej.ready_steps(sets, src, now, {500}), [2000, 3000])           # partial + later step

    def test_prompts_file(self):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "p.json")
        with open(p, "w") as f:
            json.dump({"prompts": [{"index": "5", "category": "seven hikers", "look_for": "count 7",
                                "prompt": json.dumps({"short_description": "Seven hikers on a trail."})}]}, f)
        m = ej.load_prompts({"file": p, "list_key": "prompts", "key_field": "index", "label_field": "category",
                             "text_field": "prompt", "look_for_field": "look_for", "text_chars": 100})
        self.assertEqual(m[5], {"label": "seven hikers", "text": "Seven hikers on a trail.", "look_for": "count 7"})
        shutil.rmtree(d)


class TestPolicy(unittest.TestCase):
    def d(self, h, **kw):
        return ej.decide(h, dict(POL, **kw))["decision"]

    def test_improving_continues(self):
        h = [V(500, "n/a", "n/a"), V(1000, "better", "better"), V(1500, "better", "same"), V(2000, "same", "better")]
        self.assertEqual(self.d(h), "continue")

    def test_single_regression_continues(self):
        self.assertEqual(self.d([V(500, "n/a", "n/a"), V(1000, "worse", "same")]), "continue")

    def test_regression_twice_stops(self):
        r = ej.decide([V(500, "n/a", "n/a"), V(1000, "worse", "same", regs=["#44 wheel"], prev=500),
                       V(1500, "worse", "same", regs=["persists: #44 S0 wheel"], prev=1000)], POL)
        self.assertEqual((r["decision"], r["rule"]), ("stop", "structure_regression"))
        self.assertIn("#44 wheel", r["why"])
        self.assertEqual((r["steps"], r["before_step"], r["prompts"]), ([1000, 1500], 500, [44]))
        # different prompts named -> no stop; one verdict below stop_min_confidence (0.7) -> no stop
        self.assertEqual(self.d([V(1000, "worse", regs=["#44 wheel"]), V(1500, "worse", regs=["#3 hand"])]), "continue")
        self.assertEqual(self.d([V(1000, "worse", regs=["#4"]), V(1500, "worse", regs=["#4"], conf=0.6)]), "continue")
        self.assertEqual(self.d([V(1000, "worse", regs=["#4"]), V(1500, "worse", regs=["#4"], conf=0.6)],
                                stop_min_confidence=0.5), "stop")

    def test_regression_recovered_continues(self):
        self.assertEqual(self.d([V(1000, "worse"), V(1500, "better", persist="no")]), "continue")
        self.assertEqual(self.d([V(1000, "worse"), V(1500, "better", persist="yes")]), "continue")

    def test_regression_persists_mode(self):
        h = [V(1000, "worse", regs=["#2 x"]), V(1500, "same", "better", persist="yes", regs=["persists: #2 x"])]
        self.assertEqual(self.d(h, regression_confirm="persist"), "stop")
        self.assertEqual(self.d(h, regression_confirm="worse_twice"), "continue")
        self.assertEqual(self.d([V(1000, "worse", regs=["#2"]), V(1500, "same", persist="partly", regs=["#2"])]),
                         "continue")

    def test_low_confidence_ignored(self):
        self.assertEqual(self.d([V(1000, "worse", conf=0.3), V(1500, "worse")]), "continue")

    def test_plateau(self):
        same = lambda st, **kw: V(st, "same", "same", **kw)   # noqa: E731
        # the 1,500 set itself does not count: earliest plateau stop is 2,500
        self.assertEqual(self.d([same(1000), same(1500), same(2000)]), "continue")
        r = ej.decide([same(1500), same(2000), V(2500, "same", "worse")], POL)
        self.assertEqual((r["decision"], r["rule"], r["before_step"], r["steps"]), ("stop", "plateau", 1500, [2000, 2500]))
        self.assertEqual(self.d([same(1500), V(2000, "same", "better"), same(2500)]), "continue")
        self.assertEqual(self.d([same(2500)]), "continue")                                         # only one
        self.assertEqual(self.d([same(1500, ss=3), same(2000, ss=4), same(2500, ss=4)]), "continue")   # score rose
        self.assertEqual(self.d([same(1500, ref="same"), same(2000, ref="better"), same(2500)]), "continue")  # vs ref
        self.assertEqual(self.d([same(1500), same(2000), same(2500, conf=0.65)]), "continue")

    def test_normalize(self):
        v, probs = ej.normalize({"step": 7, "structure_score": 9, "look_score": "2", "vs_previous": {"structure": "meh"},
                                 "confidence": 80, "regressions": ["x"] * 20}, 1000)
        self.assertEqual((v["step"], v["structure_score"], v["look_score"]), (1000, 5, 2))
        self.assertEqual(v["vs_previous"], {"structure": "n/a", "look": "n/a"})
        self.assertEqual((v["confidence"], len(v["regressions"]), v["recommend"]), (0.8, 12, "continue"))
        self.assertTrue(probs)


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg missing")
class TestSheet(unittest.TestCase):
    def test_build(self):
        d = tempfile.mkdtemp()
        a, b = os.path.join(d, "a.jpg"), os.path.join(d, "b.jpg")
        make_jpg(a, "red", 80)
        make_jpg(b, "blue", 40)
        r = sheet.build_sheet([("#1 two cars: label ok", [a, b, None]), ("#2", [b, None, a])], ["REF", "PREV 5", "CUR 9"],
                              os.path.join(d, "s.jpg"), title="t", tile=32)
        self.assertTrue(os.path.getsize(r["path"]) > 500)
        self.assertEqual(r["missing"], 2)
        self.assertIsNone(sheet.decode_tile(os.path.join(d, "nope.jpg"), 32))
        shutil.rmtree(d)


def fake_meter(bal):
    def f():
        return {"credits": bal[0], "primary_pct": 100}
    return f


def scripted_judge(verdicts, bal, cost=2.5, calls=None, confirm=None):
    """call_fn: returns verdicts[step]; spends `cost` credits per call. Confirmation calls (CONFIRM_SCHEMA) get
    `confirm` (default: agrees with a structure regression)."""
    def call(images, prompt, instr, jc, scratch):
        if jc.get("_schema") is ej.CONFIRM_SCHEMA:
            bal[0] -= cost
            if calls is not None:
                calls.append({"step": "confirm", "images": list(images), "prompt": prompt, "effort": jc["effort"]})
            c = confirm or {"structure": "worse", "look": "same", "regressions": ["#0 S2 extra finger"],
                            "confidence": 0.85, "reason": "worse"}
            return {"ok": True, "verdict": c, "secs": 0.1}
        step = int(prompt.split("CURRENT set = step ")[1].split(".")[0])
        if calls is not None:
            calls.append({"step": step, "images": list(images), "prompt": prompt})
        bal[0] -= cost
        v = verdicts.get(step)
        if v is None:
            return {"ok": False, "error": "boom", "secs": 0.1}
        return {"ok": True, "verdict": dict(v, step=step), "secs": 0.1, "usage": {"total_tokens": 1000}}
    return call


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg missing")
class TestRunner(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.src = os.path.join(self.d, "samples")
        for step in (500, 1000, 1500, 2000):
            for i in range(6):
                make_jpg(os.path.join(self.src, f"17930{i}__{step:09d}_{i}.jpg"), ["red", "green", "blue"][i % 3], 32)
        self.ref = os.path.join(self.d, "ref")
        make_jpg(os.path.join(self.ref, "p000.jpg"), "white", 32)
        self.shared = os.path.join(self.d, "watcher", "alerts.jsonl")
        self.marker = os.path.join(self.d, "STOPPED")

    def tearDown(self):
        shutil.rmtree(self.d)

    def cfg(self, **kw):
        c = ej.load_config(None, {
            "name": "t", "out_dir": os.path.join(self.d, "out"), "alerts_jsonl": [self.shared],
            "source": {"dir": self.src, "group_size": 3, "expect": 6, "min_age_s": 0, "settle_s": 0},
            "reference": {"dir": self.ref}, "sheet": {"variants": [0, 1], "rows_per_sheet": 1, "tile": 32},
            "act": "stop", "stop_command": {"argv": ["sh", "-c", f"echo x >> {self.marker}; echo STOP requested"],
                                            "retry_s": 0},
            **kw})
        return c

    def alerts(self, p):
        return [json.loads(x) for x in rd(p).splitlines()]

    def test_regression_stops_and_acts_once(self):
        bal, calls = [1000.0], []
        verd = {500: V(500, "n/a", "n/a"), 1000: V(1000, "worse", "same", regs=["#0 S0 extra finger"]),
                1500: V(1500, "worse", "same", regs=["persists: #0 S0 extra finger"]), 2000: V(2000, "worse", "worse")}
        j = ej.EvalJudge(self.cfg(), call_fn=scripted_judge(verd, bal, calls=calls), meter_fn=fake_meter(bal))
        msg = j.run(once=False)
        self.assertIn("stop decision handled", msg)
        self.assertEqual([c["step"] for c in calls], [500, 1000, 1500, "confirm"])   # 2000 never judged
        cf = calls[3]
        self.assertEqual(cf["effort"], "high")
        self.assertNotIn("extra finger", cf["prompt"])                         # no earlier regressions shown
        self.assertIn("BEFORE set (step 500)", cf["prompt"])
        self.assertEqual(len(cf["images"]), 1 + 2)                             # 1 zoom sheet (#0) + 2 overview
        self.assertTrue(any("/zoom/" in x for x in cf["images"]))
        self.assertEqual(rd(self.marker).count("x"), 1)               # stop command ran exactly once
        al = self.alerts(self.shared)
        verdicts = [a for a in al if a["rule"] == "eval_verdict"]
        self.assertEqual([a["severity"] for a in verdicts], ["info", "warn", "critical"])
        self.assertEqual(verdicts[-1]["decision"]["rule"], "structure_regression")
        self.assertTrue(any(a["rule"] == "eval_judge_act" and "rc=0" in a["what"] for a in al))
        self.assertFalse(any(a["rule"] == "watcher_exit" for a in al))         # exit record only in our own file
        own = self.alerts(os.path.join(self.d, "out", "alerts.jsonl"))
        self.assertEqual(own[-1]["rule"], "watcher_exit")
        log = rd(os.path.join(self.d, "out", "judge-log.md"))
        self.assertIn("## Step 1500 vs 1000", log)
        self.assertIn("ran the stop command", log)
        self.assertIn("#0 S0 extra finger", log)
        # sheets: first set has REF + CUR S0 + CUR S1; later ones add PREV columns; 2 prompts -> 2 sheets each
        self.assertEqual(len(calls[0]["images"]), 2)
        self.assertIn("PREVIOUS set = step 1000", calls[2]["prompt"])
        self.assertIn("#0 S0 extra finger", calls[2]["prompt"])                # previous regressions passed on
        st = json.loads(rd(os.path.join(self.d, "out", "state.json")))
        self.assertEqual(st["credits"]["used"], 10.0)
        self.assertEqual(st["credits"]["calls"], 4)
        # a restarted judge never acts twice
        j2 = ej.EvalJudge(self.cfg(), call_fn=scripted_judge(verd, bal), meter_fn=fake_meter(bal))
        j2.run(once=True)
        self.assertEqual(rd(self.marker).count("x"), 1)

    def test_improving_no_stop_no_act_flag(self):
        bal = [1000.0]
        verd = {500: V(500, "n/a", "n/a"), 1000: V(1000, "better", "better"), 1500: V(1500, "better", "same"),
                2000: V(2000, "same", "better")}
        j = ej.EvalJudge(self.cfg(lifecycle={"last_step": 2000}), call_fn=scripted_judge(verd, bal),
                         meter_fn=fake_meter(bal))
        self.assertIn("last step 2000 judged", j.run())
        self.assertFalse(os.path.exists(self.marker))
        self.assertEqual(sorted(int(k) for k in j.st["judged"]), [500, 1000, 1500, 2000])

    def test_no_act_recommends_only(self):
        bal = [1000.0]
        verd = {s: V(s, "same", "same") for s in (500, 1000, 1500, 2000)}
        j = ej.EvalJudge(self.cfg(policy={"plateau_min_step": 500}, lifecycle={"last_step": 2000}), no_act=True, call_fn=scripted_judge(verd, bal),
                         meter_fn=fake_meter(bal))
        j.run()
        self.assertFalse(os.path.exists(self.marker))
        self.assertIsNone(j.st.get("acted"))
        self.assertEqual(sorted(int(k) for k in j.st["judged"]), [500, 1000, 1500, 2000])   # keeps judging
        self.assertEqual([x["step"] for x in j.st["recommended_stop"]], [1500, 2000])
        crit = [a for a in self.alerts(self.shared) if a["severity"] == "critical"]
        self.assertEqual(crit[0]["decision"]["rule"], "plateau")
        self.assertIn("act=none", crit[0]["action"])

    def test_credit_cap(self):
        bal = [1000.0]
        verd = {s: V(s, "better", "better") for s in (500, 1000, 1500, 2000)}
        j = ej.EvalJudge(self.cfg(judge={"credit_cap": 6}), call_fn=scripted_judge(verd, bal, cost=2.5),
                         meter_fn=fake_meter(bal))
        j.run(once=True)
        self.assertEqual(sorted(int(k) for k in j.st["judged"]), [500, 1000])  # 5.0 used, +2.5 would pass 6
        caps = [a for a in self.alerts(self.shared) if a["rule"] == "eval_judge_credit_cap"]
        self.assertEqual(len(caps), 1)

    def test_call_failure_alerts_and_moves_on(self):
        bal = [1000.0]
        verd = {500: V(500, "n/a", "n/a"), 1500: V(1500, "better", "better"), 2000: V(2000, "same", "same")}
        j = ej.EvalJudge(self.cfg(), call_fn=scripted_judge(verd, bal), meter_fn=fake_meter(bal))
        j.run(once=True)
        self.assertIn("1000", j.st["failed"])
        self.assertEqual(j.st["credits"]["calls"], 5)                          # 1000 tried twice (retries 1)
        errs = [a for a in self.alerts(self.shared) if a["rule"] == "eval_judge_error"]
        self.assertEqual(len(errs), 1)
        last = json.loads(rd(os.path.join(self.d, "out", "verdicts.jsonl")).splitlines()[-1])
        self.assertEqual(last["prev_step"], 1500)

    def test_fetch_ssh_with_local_shell(self):
        dest = os.path.join(self.d, "cache")
        rels = sorted(os.listdir(self.src))[:3]
        n = ej.fetch_ssh(["sh", "-c"], self.src, rels, dest)
        self.assertEqual(n, 3)
        self.assertEqual(sorted(os.listdir(dest)), rels)


class TestLaunch(unittest.TestCase):
    PAT = r"^(?P<ts>\d+)__(?P<step>\d+)_(?P<idx>\d+)\.jpg$"

    def files(self, t0, steps, n=3):
        return [(f"{int((t0 + k * 60 + i) * 1000)}__{st:09d}_{i}.jpg", 10, 0.0) for k, st in enumerate(steps)
                for i in range(n)]

    def test_restart_from_zero_drops_old_launch(self):
        src = dict(ej.DEFAULTS["source"], pattern=self.PAT, group_size=1)
        t0 = 1_793_000_000
        old = self.files(t0, [500, 1000])
        new = self.files(t0 + 4 * 3600, [500])
        sets, launch = ej.group_sets(old + new, src, with_launch=True)
        self.assertEqual(sorted(sets), [500])
        self.assertTrue(all(r.startswith(str(int((t0 + 4 * 3600) * 1000))[:6]) for r in sets[500]["files"].values()))
        self.assertAlmostEqual(launch, t0 + 4 * 3600, delta=1)
        # min_epoch: ignore everything before it
        sets = ej.group_sets(old + new, dict(src, min_epoch=t0 + 3600))
        self.assertEqual(sorted(sets), [500])
        # same step only, old launch never got past 500
        sets, launch2 = ej.group_sets(self.files(t0, [500]) + new, src, with_launch=True)
        self.assertEqual(launch2, launch)

    def test_resume_keeps_earlier_steps(self):
        src = dict(ej.DEFAULTS["source"], pattern=self.PAT, group_size=1)
        t0 = 1_793_000_000
        a = self.files(t0, [500, 1000, 1500])
        b = self.files(t0 + 5 * 3600, [1000, 1500])          # resume from 750 re-renders 1000, 1500
        sets, launch = ej.group_sets(a + b, src, with_launch=True)
        self.assertEqual(sorted(sets), [500, 1000, 1500])
        self.assertAlmostEqual(launch, t0, delta=1)
        self.assertGreater(sets[1000]["newest"], t0 + 5 * 3600)


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg missing")
class TestFixes(TestRunner):
    """Review fixes: stop retries, truncated images, partial sets, confirmation veto, new launch, cache, gates."""

    def test_stop_command_retried_until_ok(self):
        cnt = os.path.join(self.d, "cnt")
        argv = ["sh", "-c", f"echo x >> {cnt}; n=$(wc -l < {cnt}); [ $n -ge 3 ] && echo ok || {{ echo refused; exit 255; }}"]
        bal = [1000.0]
        verd = {500: V(500, "n/a", "n/a"), 1000: V(1000, "worse", regs=["#1 hand"]),
                1500: V(1500, "worse", regs=["#1 hand"]), 2000: V(2000, "same")}
        j = ej.EvalJudge(self.cfg(stop_command={"argv": argv, "retry_s": 0, "attempts": 3,
                                                "followup_check": {"argv": ["false"], "missing_msg": "NO FINISH LOOP"}}),
                         call_fn=scripted_judge(verd, bal), meter_fn=fake_meter(bal))
        self.assertIn("stop decision handled", j.run())
        self.assertEqual(rd(cnt).count("x"), 3)
        self.assertEqual(j.st["acted"]["attempts"], 3)
        ran = [a for a in self.alerts(self.shared) if a["rule"] == "eval_judge_act"]
        self.assertEqual(len(ran), 1)
        self.assertIn("NO FINISH LOOP", ran[0]["action"])

    def test_stop_command_all_fail_keeps_judging_then_retries(self):
        flag = os.path.join(self.d, "box-up")
        argv = ["sh", "-c", f"[ -e {flag} ] && echo ok || {{ echo 'Connection refused'; exit 255; }}"]
        bal = [1000.0]
        verd = {500: V(500, "n/a", "n/a"), 1000: V(1000, "worse", regs=["#1 hand"]),
                1500: V(1500, "worse", regs=["#1 hand"]), 2000: V(2000, "same")}
        j = ej.EvalJudge(self.cfg(stop_command={"argv": argv, "retry_s": 0, "attempts": 3}),
                         call_fn=scripted_judge(verd, bal), meter_fn=fake_meter(bal))
        j.run(once=True)
        self.assertIsNone(j.st.get("acted"))
        self.assertEqual(j.st["pending_stop"]["attempts"], 3)
        self.assertIn("2000", j.st["judged"])                                # kept judging
        crit = [a for a in self.alerts(self.shared) if a["rule"] == "eval_judge_act"]
        self.assertEqual(len(crit), 1)
        self.assertIn("FAILED", crit[0]["what"])
        open(flag, "w").close()                                               # the box comes back
        j2 = ej.EvalJudge(self.cfg(stop_command={"argv": argv, "retry_s": 0}), call_fn=scripted_judge(verd, bal),
                          meter_fn=fake_meter(bal))
        self.assertIn("stop decision handled", j2.run())
        self.assertEqual(j2.st["acted"]["attempts"], 4)

    def test_confirmation_veto_continues(self):
        bal, calls = [1000.0], []
        verd = {500: V(500, "n/a", "n/a"), 1000: V(1000, "worse", regs=["#1 hand"]),
                1500: V(1500, "worse", regs=["#1 hand"]), 2000: V(2000, "same")}
        j = ej.EvalJudge(self.cfg(lifecycle={"last_step": 2000}),
                         call_fn=scripted_judge(verd, bal, calls=calls, confirm={
                             "structure": "same", "look": "same", "regressions": [], "confidence": 0.8,
                             "reason": "seed noise"}), meter_fn=fake_meter(bal))
        self.assertIn("last step 2000 judged", j.run())
        self.assertFalse(os.path.exists(self.marker))
        self.assertEqual([c["step"] for c in calls], [500, 1000, 1500, "confirm", 2000])
        self.assertFalse(j.st["confirmations"][0]["agree"])
        self.assertTrue(any(a["rule"] == "eval_judge_confirm" for a in self.alerts(self.shared)))
        self.assertIn("confirmation check disagreed", rd(os.path.join(self.d, "out", "judge-log.md")))

    def test_truncated_image_waits_then_partial_is_recommend_only(self):
        p = os.path.join(self.src, f"179301__{1500:09d}_1.jpg")
        with open(p, "rb") as f:
            data = f.read()
        with open(p, "wb") as f:
            f.write(data[:len(data) // 2])                                    # copied mid-write
        self.assertFalse(ej.image_complete(p))
        bal = [1000.0]
        verd = {500: V(500, "n/a", "n/a"), 1000: V(1000, "worse", regs=["#1 hand"]),
                1500: V(1500, "worse", regs=["#1 hand"]), 2000: V(2000, "worse", regs=["#1 hand"])}
        src = dict(self.cfg()["source"], launch_gap_s=0)                       # we back-date mtimes below
        j = ej.EvalJudge(self.cfg(source=src), call_fn=scripted_judge(verd, bal), meter_fn=fake_meter(bal))
        j.tick()
        self.assertNotIn("1500", j.st["judged"])                              # waits for rsync to replace it
        self.assertNotIn("2000", j.st["judged"])                              # later sets wait (step order)
        # still truncated long after: judged without it, marked partial, a stop on it is recommend-only
        old = time.time() - 4000
        for f in os.listdir(self.src):
            if f"_{1500:09d}_" in f:
                os.utime(os.path.join(self.src, f), (old, old))
        j.tick()
        self.assertIn("1500", j.st["judged"])
        h = {x["step"]: x for x in j.st["history"]}
        self.assertTrue(h[1500]["partial"])
        self.assertIn("2000", j.st["judged"])
        self.assertEqual([d["decision"] for d in j.st["decisions"]], ["continue", "continue", "stop", "stop"])
        self.assertEqual(len(j.st["recommended_stop"]), 2)                    # stop on a partial set: recommend only
        self.assertFalse(os.path.exists(self.marker))

    def test_new_launch_resets_state(self):
        bal = [1000.0]
        pat = r"^(?P<ts>\d+)__(?P<step>\d+)_(?P<idx>\d+)\.jpg$"
        src2 = os.path.join(self.d, "s2")
        t0 = time.time() - 6 * 3600
        for k, step in enumerate((500, 1000)):
            for i in range(6):
                make_jpg(os.path.join(src2, f"{int((t0 + k * 600 + i) * 1000)}__{step:09d}_{i}.jpg"), "red", 32)
        verd = {500: V(500, "n/a", "n/a"), 1000: V(1000, "better", "better")}
        cfg = self.cfg(source={"dir": src2, "pattern": pat, "group_size": 3, "expect": 6, "min_age_s": 0})
        j = ej.EvalJudge(cfg, call_fn=scripted_judge(verd, bal), meter_fn=fake_meter(bal))
        j.tick()
        self.assertEqual(sorted(j.st["judged"]), ["1000", "500"])
        t1 = time.time() - 600                                                # relaunch from update 0
        for i in range(6):
            make_jpg(os.path.join(src2, f"{int((t1 + i) * 1000)}__{500:09d}_{i}.jpg"), "blue", 32)
        j.tick()
        self.assertEqual(sorted(j.st["judged"]), ["500"])
        self.assertIsNone(j.st["history"][0]["prev_step"])
        self.assertTrue(any(a["rule"] == "eval_judge_launch" for a in self.alerts(self.shared)))
        self.assertTrue(glob_one(os.path.join(self.d, "out", "state-launch-*.json")))

    def test_cache_refetch_on_change(self):
        cfg = self.cfg(source={"ssh": ["sh", "-c"], "dir": self.src, "group_size": 3, "expect": 6})
        j = ej.EvalJudge(cfg, call_fn=None, meter_fn=fake_meter([0.0]))
        j.list_sets = lambda: ej.group_sets(ej.list_local(self.src), cfg["source"])   # BSD find has no -printf
        sets = j.list_sets()
        paths = j.local_paths(500, sets[500])
        k = sorted(paths)[0]
        size0 = os.path.getsize(paths[k])
        rel = sets[500]["files"][k]
        make_jpg(os.path.join(self.src, rel), "yellow", 96)                   # rsync replaced the file
        os.utime(os.path.join(self.src, rel), (time.time() + 5, time.time() + 5))
        sets = j.list_sets()
        paths = j.local_paths(500, sets[500])
        self.assertNotEqual(os.path.getsize(paths[k]), size0)

    def test_gate_outputs_logged(self):
        g = os.path.join(self.d, "gate", "250")
        os.makedirs(g)
        with open(os.path.join(g, "GATE.json"), "w") as f:
            json.dump({"step": 250, "summary": {"n": 72, "ft_vs_base_edge": 0.61, "drift_ratio_vs_F": 1.2},
                       "alerts": []}, f)
        make_jpg(os.path.join(g, "sheet-1.jpg"), "red", 32)
        make_jpg(os.path.join(g, "sheet-2.jpg"), "red", 32)
        old = time.time() - 300
        for f in os.listdir(g):
            os.utime(os.path.join(g, f), (old, old))
        os.makedirs(os.path.join(self.d, "gate", "base"))
        bal = [1000.0]
        j = ej.EvalJudge(self.cfg(gates={"dir": os.path.join(self.d, "gate")}),
                         call_fn=scripted_judge({500: V(500, "n/a", "n/a")}, bal), meter_fn=fake_meter(bal))
        j.gates_tick()
        j.gates_tick()
        al = [a for a in self.alerts(self.shared) if a["rule"] == "eval_gate"]
        self.assertEqual(len(al), 1)
        self.assertIn("drift vs F 1.2", al[0]["what"])
        self.assertIn("## Early-stop gate 250", rd(os.path.join(self.d, "out", "judge-log.md")))


def glob_one(p):
    import glob
    return glob.glob(p)


if __name__ == "__main__":
    unittest.main()
