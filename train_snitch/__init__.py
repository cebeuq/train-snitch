"""train-snitch: a cheap, read-only watcher for ML training runs that snitches to a bigger agent.

Deterministic probes + rules run every cycle for free; a small LLM judge (through the Pi harness) is
called only on anomalies, state changes and a slow periodic digest, under a hard call/credit budget.
"""
__version__ = "0.1.0"
