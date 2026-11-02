#!/usr/bin/env python3
"""Regenerate the three synthetic `generic` incidents (HF Trainer + torchrun crash, a quiet Lightning run, a hung
DeepSpeed rank). Each cycle stores the snapshot the generic adapter's collect() would produce: the text log lines
plus the metric rows parsed from them by train_snitch.parsers (so the replays also exercise the parsers).
Run: python3 -I tests/replay/tools/make_generic_incidents.py"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPLAY = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(REPLAY)))
from train_snitch import parsers  # noqa: E402

T0 = 1793502000  # Oct 31 2026 07:20 PM PDT


def snap(i, lines=None, fmt="auto", alive=True, gpus=(95, 96), disk=400.0, ckpt_age_min=10, ctr=None):
    t = T0 + 60 * i
    lines = lines or []
    rows = parsers.log_rows(lines, fmt, counter=ctr)
    s = {"now": t, "log_lines": {"trainer": lines} if lines else {}, "metric_rows": {"trainer": rows},
         "processes": {"proc": {"alive": alive, "count": int(alive), "sample": ["4242 torchrun --nproc_per_node 2 train.py"] if alive else []}},
         "gpu": [[t, k, float(u), 300.0 if u > 10 else 80.0, f"{k}, {u}, 300, 150000, 183000, 60"] for k, u in enumerate(gpus)],
         "disk": {"disk": disk},
         "checkpoints": {"ckpt": {"count": 3, "newest": "/work/run/checkpoint-300", "newest_mtime": t - 60 * ckpt_age_min,
                                  "stale_min": 90, "now": t}},
         "source_errors": {}}
    return {"t": t, "t_local": f"cycle {i}", "signals": s}


def hf(loss, gn, ep):
    return f"{{'loss': {loss}, 'grad_norm': {gn}, 'learning_rate': 1.8e-05, 'epoch': {ep}}}"


def write(name, title, timeline, expected, notes):
    d = os.path.join(REPLAY, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "incident.json"), "w") as f:
        json.dump({"id": name, "title": title, "source": "synthetic", "adapter": "generic",
                   "provenance": ["synthetic: built by tests/replay/tools/make_generic_incidents.py in the frameworks' real log formats"],
                   "notes": notes}, f, indent=1)
    with open(os.path.join(d, "timeline.jsonl"), "w") as f:
        for row in timeline:
            f.write(json.dumps(row) + "\n")
    with open(os.path.join(d, "expected.json"), "w") as f:
        json.dump(expected, f, indent=1)
    return {"id": name, "source": "synthetic", "title": title, "worst": max((v["severity"] for v in expected["verdicts"]),
            key=["info", "warn", "critical"].index, default="none"), "cycles": len(timeline)}


def main():
    out = []
    # 1. HF Trainer under torchrun: grad norm goes NaN, then the loss, then rank 0 dies and torchrun tears down
    c = {"n": 0}
    tl = [snap(0, [hf(0.5121, 1.31, 0.40), hf(0.5034, 1.22, 0.41)], ctr=c),
          snap(1, [hf(0.4987, 1.40, 0.42), hf(0.5102, 1.35, 0.43)], ctr=c),
          snap(2, [hf(0.6120, "nan", 0.44)], ctr=c),
          snap(3, [hf("nan", "nan", 0.45), hf("nan", "nan", 0.46)], ctr=c),
          snap(4, ["Traceback (most recent call last):", '  File "train.py", line 212, in <module>',
                   "RuntimeError: Function 'MulBackward0' returned nan values in its 0th output.",
                   "E1031 19:24:41.123000 4242 torch/distributed/elastic/multiprocessing/api.py:869] failed (exitcode: 1) local_rank: 0 (pid: 4250) of binary: /usr/bin/python3",
                   "torch.distributed.elastic.multiprocessing.errors.ChildFailedError: "], alive=True, gpus=(0, 0), ctr=c),
          snap(5, [], alive=False, gpus=(0, 0), ctr=c),
          snap(6, [], alive=False, gpus=(0, 0), ctr=c)]
    out.append(write("generic-hf-nan-torchrun-crash", "HF Trainer under torchrun: grad_norm NaN, then loss NaN, then ChildFailedError and the process is gone", tl, {
        "verdicts": [
            {"cycle": 2, "severity": "warn", "rule": "log:grad_nonfinite", "status": "new"},
            {"cycle": 3, "severity": "critical", "rule": "loss_nonfinite", "status": "new", "what_contains": ["NaN"]},
            {"cycle": 4, "severity": "critical", "rule": "log:worker_failed", "status": "new", "evidence_contains": ["exitcode: 1"]},
            {"cycle": 6, "severity": "critical", "rule": "process_gone", "status": "new", "what_contains": ["proc"]}],
        "must_not": [{"cycle": 0, "severity_at_least": "warn"}, {"cycle": 1, "severity_at_least": "warn"}],
        "max_alerts_warn_or_worse": 6}, "process_gone needs 2 missed checks (min_cycles) so a flaky ps never pages."))
    # 2. Quiet Lightning run with real-world noise: must not alert
    c = {"n": 0}
    tl = []
    for i in range(10):
        base = 450 + 40 * i
        lines = [f"Epoch 1:  {base // 10}%|####      | {base + k * 10}/1000 [01:02<01:10,  7.2it/s, loss={0.62 - 0.004 * i - 0.001 * k:.3f}, v_num=3]"
                 for k in range(4)]
        if i == 0:
            lines += ["[rank1]:[W1031 19:20:01.000 ProcessGroupNCCL.cpp:4115] Warning: An unbatched P2P op (send/recv) was called on this ProcessGroup",
                      "Gradient overflow.  Skipping step, loss scale 65536 -> 32768", "Gradient overflow.  Skipping step, loss scale 32768 -> 16384",
                      "Gradient overflow.  Skipping step, loss scale 16384 -> 8192"]
        if i == 6:
            lines += ["Epoch 1, global step 700: 'val_loss' reached 0.58 (best 0.58), saving model to '/work/run/checkpoint-700.ckpt' as top 1"]
        tl.append(snap(i, lines, ctr=c, gpus=(97, 96) if i != 6 else (8, 9), ckpt_age_min=10 + 2 * i if i < 6 else 1))
    out.append(write("generic-lightning-healthy", "NEGATIVE: healthy Lightning run with startup loss-scale overflows, a benign NCCL warning, a save pause", tl, {
        "verdicts": [], "must_not": [{"severity_at_least": "warn"}], "max_alerts_warn_or_worse": 0},
        "3 fp16 overflows at start stay under min_count 5; one GPU dip during a save is not 'idle'."))
    # 3. DeepSpeed rank hangs: rank 1 GPU drops to 0 %, rows stop, then the NCCL watchdog times out
    c = {"n": 0}
    ds = lambda it, loss: f" iteration {it:>8}/   20000 | consumed samples: {it * 32} | elapsed time per iteration (ms): 812.3 | lm loss: {loss:.6E} | loss scale: 65536.0 |"  # noqa: E731
    tl = [snap(0, [ds(1200, 2.31), ds(1210, 2.30)], ctr=c, gpus=(99, 98)),
          snap(1, [ds(1220, 2.29), ds(1230, 2.29)], ctr=c, gpus=(99, 98)),
          snap(2, [], ctr=c, gpus=(100, 0)),
          snap(3, [], ctr=c, gpus=(100, 0)),
          snap(4, ["[rank0]:[E1031 19:34:02.551 ProcessGroupNCCL.cpp:616] [Rank 0] Watchdog caught collective operation timeout: "
                   "WorkNCCL(SeqNum=40213, OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=600000) ran for 600012 milliseconds before timing out."],
               ctr=c, gpus=(100, 0))]
    out.append(write("generic-deepspeed-rank-hung", "DeepSpeed rank 1 hangs: its GPU drops to 0 % while rank 0 spins at 100 %, then the NCCL watchdog fires", tl, {
        "verdicts": [{"cycle": 3, "severity": "warn", "rule": "gpu_util_drop", "status": "new", "what_contains": ["GPU 1"], "escalate_to_judge": True},
                     {"cycle": 4, "severity": "critical", "rule": "log:nccl_timeout", "status": "new"}],
        "must_not": [{"cycle": 0, "severity_at_least": "warn"}, {"cycle": 1, "severity_at_least": "warn"}],
        "max_alerts_warn_or_worse": 3}, "Rows stop at cycle 2; the stall rule waits 15 min, the GPU asymmetry is the earlier signal."))
    idx_p = os.path.join(REPLAY, "INDEX.json")
    idx = [x for x in json.load(open(idx_p)) if not x["id"].startswith("generic-")] + out
    with open(idx_p, "w") as f:
        json.dump(sorted(idx, key=lambda x: x["id"]), f, indent=1)
    print("wrote", [o["id"] for o in out])


if __name__ == "__main__":
    main()
