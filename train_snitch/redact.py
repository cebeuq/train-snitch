"""Secret redaction. Training and hub logs sometimes print tokens; nothing token-like may leave the watcher
(LLM judge, alerts, webhook, state files). Applied at ingest and again before any channel / judge output.
"""
from __future__ import annotations

import re

MASK = "[REDACTED]"

_PATTERNS = [
    # PEM private keys (multi-line or flattened)
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)", re.S), MASK),
    # provider tokens with a recognizable prefix
    (re.compile(r"\bhf_[A-Za-z0-9]{20,}"), MASK),
    (re.compile(r"\bsk-(?:proj-|ant-|or-)?[A-Za-z0-9_\-]{20,}"), MASK),
    (re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}"), MASK),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"), MASK),
    (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), MASK),
    (re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}"), MASK),
    (re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{5,}"), MASK),   # JWT
    # Authorization headers
    (re.compile(r"\b([Bb]earer|BEARER|Basic|Token)\s+[A-Za-z0-9._~+/=\-]{12,}"), r"\1 " + MASK),
    # key=value / key: value / "key": "value" for secret-ish names (also URL query params like ?token=...)
    (re.compile(r"(?i)((?:api[_-]?key|access[_-]?key|secret[_-]?key|secret|token|passw(?:or)?d|passwd|auth|credential|"
                r"aws_secret_access_key|wandb_api_key|hf_token|private[_-]?key)[\"']?\s*[:=]\s*[\"']?)([^\s\"'&,;}]{6,})"),
     r"\1" + MASK),
    # W&B API keys: 40 hex chars near "wandb"
    (re.compile(r"(?i)(wandb[^\n]{0,40}?)\b[0-9a-f]{40}\b"), r"\1" + MASK),
    # credentials embedded in URLs: scheme://user:pass@host
    (re.compile(r"(\b[a-z][a-z0-9+.\-]*://[^\s:/@]+:)[^\s@/]{3,}@"), r"\1" + MASK + "@"),
]


def redact(s):
    """Mask secrets in a string; other types pass through unchanged."""
    if not isinstance(s, str) or not s:
        return s
    for rx, rep in _PATTERNS:
        s = rx.sub(rep, s)
    return s


def deep(x):
    """Redact every string inside a JSON-like structure (dict keys are left alone)."""
    if isinstance(x, str):
        return redact(x)
    if isinstance(x, dict):
        return {k: deep(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [deep(v) for v in x]
    return x


def _exec_filtered(argv: list[str]) -> int:
    """`python3 -I redact.py -- CMD ...`: run CMD, print its stdout/stderr redacted, exit with its rc.
    The LLM judge's read-only probes are wrapped in this so their output is masked before the model sees it."""
    import subprocess
    import sys
    try:
        cp = subprocess.run(argv, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120)
    except subprocess.TimeoutExpired:
        sys.stderr.write("timeout\n")
        return 124
    except OSError as e:
        sys.stderr.write(f"{e}\n")
        return 127
    sys.stdout.write(redact(cp.stdout or ""))
    sys.stderr.write(redact(cp.stderr or ""))
    return cp.returncode


if __name__ == "__main__":
    import sys
    a = sys.argv[1:]
    if a[:1] == ["--"]:
        a = a[1:]
    if a:
        raise SystemExit(_exec_filtered(a))
    sys.stdout.write(redact(sys.stdin.read()))
