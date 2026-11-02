#!/usr/bin/env python3
"""Copy the canonical skill (integrations/skill/train-snitch/SKILL.md) into every harness package, and its body
into gemini-cli/GEMINI.md. Plugin installs copy files, so the copies are real files, not symlinks."""
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
CANON = os.path.join(HERE, "skill", "train-snitch", "SKILL.md")
TARGETS = ["claude-code", "codex", "cursor", "copilot", "antigravity", "gemini-cli", "opencode"]

for t in TARGETS:
    d = os.path.join(HERE, t, "skills", "train-snitch")
    os.makedirs(d, exist_ok=True)
    shutil.copyfile(CANON, os.path.join(d, "SKILL.md"))
with open(CANON) as f:
    body = f.read().split("---\n", 2)[2]
with open(os.path.join(HERE, "gemini-cli", "GEMINI.md"), "w") as f:
    f.write(body)
print("synced", len(TARGETS), "copies")
