"""Replay every recorded incident and check the alerts against its expected.json (see tests/replay/README.md)."""
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from train_snitch import config as cfgmod  # noqa: E402
from train_snitch.engine import SEV  # noqa: E402
from train_snitch.replay import replay  # noqa: E402

REPLAY = os.path.join(ROOT, "tests", "replay")
CONFIGS = {"generic": os.path.join(REPLAY, "generic-sources.config.json")}
LOUD = ("new", "escalated", "worsened")


def jload(p):
    with open(p) as f:
        return json.load(f)


def rule_ok(want, got):
    return want is None or got in want.split("|")


def run_incident(name):
    inc = jload(os.path.join(REPLAY, name, "incident.json"))
    cfg = cfgmod.load(CONFIGS[inc["adapter"]], {"judge": {"enabled": False}})
    return replay(os.path.join(REPLAY, name), cfg), cfg


class ReplayTests(unittest.TestCase):
    maxDiff = None

    def check(self, name):
        cycles, cfg = run_incident(name)
        exp = jload(os.path.join(REPLAY, name, "expected.json"))
        log = "\n".join(f"[{i}] {r['severity']} {r['status']} {r['rule']}: {r['what']}"
                        for i, c in enumerate(cycles) for r in c["records"])
        for v in exp.get("verdicts", []):
            window = v.get("cycle_window") or [v["cycle"], v["cycle"]]
            hits = []
            for c in range(window[0], window[1] + 1):
                if v["status"] == "ongoing":
                    act = [a for a in cycles[c]["active"] if a["severity"] == v["severity"] and rule_ok(v.get("rule"), a["rule"])]
                    if act:
                        hits.append(({"what": act[0]["what"], "evidence": [], "action": ""}, c))
                    if v.get("max_new_alerts") == 0:
                        new = [r for r in cycles[c]["records"] if r["status"] in LOUD and SEV[r["severity"]] >= 1]
                        self.assertEqual(new, [], f"{name} cycle {c}: unexpected new alerts\n{log}")
                    continue
                hits += [(r, c) for r in cycles[c]["records"] if r["status"] == v["status"]
                         and r["severity"] == v["severity"] and rule_ok(v.get("rule"), r["rule"])]
            self.assertTrue(hits, f"{name}: missing {v}\n{log}")
            r = hits[0][0]
            for s in v.get("what_contains", []):
                self.assertIn(s, r["what"], f"{name}: what")
            for s in v.get("evidence_contains", []):
                self.assertIn(s, "\n".join(r["evidence"]), f"{name}: evidence of {r['rule']}: {r['evidence']}")
            for s in v.get("suggested_action_contains", []):
                self.assertIn(s, r["action"], f"{name}: action")
            if v.get("escalate_to_judge"):
                self.assertTrue(r.get("escalate_to_judge"), f"{name}: {r['rule']} should ask the judge")
        for m in exp.get("must_not", []):
            for i, c in enumerate(cycles):
                if "cycle" in m and m["cycle"] != i:
                    continue
                bad = [r for r in c["records"] if r["status"] != "resolved" and SEV[r["severity"]] >= SEV[m["severity_at_least"]]
                       and ("rule" not in m or r["rule"] == m["rule"])]
                self.assertEqual(bad, [], f"{name}: must_not {m} violated at cycle {i}\n{log}")
        loud = [r for c in cycles for r in c["records"] if r["status"] in LOUD and SEV[r["severity"]] >= SEV["warn"]]
        self.assertLessEqual(len(loud), exp.get("max_alerts_warn_or_worse", 99), f"{name}: too many alerts\n{log}")
        # dedupe: reminders for one key are bounded by the reminder interval
        span = cycles[-1]["t"] - cycles[0]["t"]
        e = cfg["engine"]
        for key in {r["key"] for c in cycles for r in c["records"]}:
            rem = [r for c in cycles for r in c["records"] if r["key"] == key and r["status"] == "ongoing"]
            if rem:
                gap = e["remind_critical_s"] if rem[0]["severity"] == "critical" else e["remind_warn_s"]
                self.assertLessEqual(len(rem), span // gap + 1, f"{name}: {key} reminders spam")


def _make(name):
    return lambda self: self.check(name)


for _n in jload(os.path.join(REPLAY, "INDEX.json")):
    setattr(ReplayTests, "test_" + _n["id"].replace("-", "_"), _make(_n["id"]))

if __name__ == "__main__":
    unittest.main()
