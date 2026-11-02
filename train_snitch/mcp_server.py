"""MCP stdio server: lets any MCP-capable agent harness (Claude Code, Codex, Cursor, Copilot, Gemini / Antigravity,
OpenCode, ...) start, stop, list and read train-snitch watchers.

Built on the official MCP Python SDK (optional extra: pip install 'train-snitch[mcp]'); the tools are thin
wrappers over `control.py`, which holds the safety boundary (allowlisted folders, watcher-only SIGTERM, no shell).

  train-snitch-mcp            (console script)    or    python3 -m train_snitch mcp
Environment: TRAIN_SNITCH_CONFIG_DIRS, TRAIN_SNITCH_OUT_DIRS, TRAIN_SNITCH_ENV_FILE (see docs/HARNESSES.md).
"""
from __future__ import annotations

from typing import Any

from . import __version__, control

INSTRUCTIONS = """train-snitch is a read-only watchdog for ML training runs. A watcher reads logs, GPUs, processes and
billing every minute, runs deterministic rules and a small LLM judge, and writes alerts. It never changes the run.
Use list_watches first. start_watch starts a finite, detached watcher for a config file; stop_watch stops only the
watcher, never the training run. Poll tail_alerts with since=next_since to follow alerts; report critical alerts to
the user at once, quoting `what` and `evidence`. Never stop, kill or destroy a training run or cloud instance
because of an alert: recommend it and let the user decide."""


def build_server():
    try:
        from mcp.server.mcpserver import MCPServer
        from mcp.server.mcpserver.exceptions import ToolError
        from mcp.types import ToolAnnotations
    except ImportError as e:   # pragma: no cover - depends on the optional extra
        raise SystemExit("the MCP server needs the official MCP Python SDK 2.x: pip install 'train-snitch[mcp]' "
                         f"({e})") from None

    server = MCPServer("train-snitch", version=__version__, instructions=INSTRUCTIONS)
    ro = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)

    def call(fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except control.ControlError as e:
            raise ToolError(str(e)) from None

    @server.tool(annotations=ro)
    def list_watches() -> dict[str, Any]:
        """List train-snitch watchers under the allowed output folders: run name, out_dir, alive, pid, last update,
        deadline, judge mode, number of active alerts and the worst active severity."""
        return call(control.list_watches)

    @server.tool(annotations=ro)
    def status(run: str) -> dict[str, Any]:
        """Heartbeat of one watcher (status.json): active alerts, the judge's last verdict, budget, deadline.
        `run` is an out_dir path or a run name from list_watches."""
        return call(control.status, run)

    @server.tool(annotations=ro)
    def latest(run: str) -> dict[str, Any]:
        """latest.txt of one watcher: the newest warn+ alert in plain text plus every still-active problem."""
        return call(control.latest, run)

    @server.tool(annotations=ro)
    def tail_alerts(run: str, n: int = 20, min_severity: str = "warn", since: float | None = None,
                    include_ongoing: bool = False) -> dict[str, Any]:
        """Newest alert records (alerts.jsonl), filtered like `train-snitch tail`: severity >= min_severity
        (info|warn|critical), plus judge false-alarm notes, eval-judge records, TEST records and start/exit records.
        Without `since`: the newest n. To follow a watcher, call again with since=<next_since from the previous
        call>: you then get the oldest n newer records, and `more: true` means call again at once."""
        return call(control.tail_alerts, run, n=n, min_severity=min_severity, since=since, include_ongoing=include_ongoing)

    @server.tool(annotations=ro)
    def check_config(config: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        """Validate a run config (path under TRAIN_SNITCH_CONFIG_DIRS) without running anything: errors, warnings,
        out_dir and the resolved judge model chain (provider, endpoint, auth env var NAME, reasoning)."""
        return call(control.check_config, config, overrides)

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False,
                                             openWorldHint=True))
    def start_watch(config: str, overrides: dict[str, Any] | None = None, rules_only: bool = False, dry: bool = False,
                    once: bool = False, test_alert: bool = False) -> dict[str, Any]:
        """Start a detached, finite watcher for a run config (a .json path under TRAIN_SNITCH_CONFIG_DIRS). It exits
        by itself when the run ends or at its deadline. `overrides`: small inline changes only (name, cycle_s,
        timezone, lifecycle deadline keys, rules thresholds, judge model / fallbacks / limits). rules_only: no LLM.
        dry: no desktop notification / webhook / command hook. once: one cycle. test_alert: one TEST record
        through every channel."""
        return call(control.start_watch, config, overrides, rules_only=rules_only, dry=dry, once=once,
                    test_alert=test_alert)

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True,
                                             openWorldHint=False))
    def stop_watch(run: str) -> dict[str, Any]:
        """Stop one watcher (SIGTERM to the pid in its out_dir, only if that process is a train-snitch watcher).
        It writes its exit record first. This never touches the training run itself."""
        return call(control.stop_watch, run)

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True,
                                             openWorldHint=True))
    def meter() -> dict[str, Any]:
        """Codex usage meter and credit balance (reads the local Codex CLI login; no model call). Only relevant
        when the judge uses an openai-codex model."""
        return call(control.meter)

    return server


def main() -> None:
    build_server().run()   # stdio transport (the SDK default)


if __name__ == "__main__":
    main()
