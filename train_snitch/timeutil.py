"""Time helpers. All human-facing times are 12-hour clock with AM/PM in the configured time zone."""
from __future__ import annotations

import re
import time
from datetime import datetime
from zoneinfo import ZoneInfo

_TZ = None


def set_tz(name: str | None) -> None:
    global _TZ
    _TZ = ZoneInfo(name) if name else None


def tz():
    return _TZ


def _dt(t: float) -> datetime:
    return datetime.fromtimestamp(t, _TZ) if _TZ else datetime.fromtimestamp(t).astimezone()


def fmt12(t: float | None, date: bool = False, seconds: bool = False) -> str:
    """'07:32 PM PDT' (or 'Oct 31 07:32 PM PDT' with date=True)."""
    if t is None:
        return "?"
    d = _dt(t)
    f = ("%b %d " if date else "") + ("%I:%M:%S %p" if seconds else "%I:%M %p")
    return (d.strftime(f) + " " + (d.tzname() or "")).strip()


def stamp(t: float) -> str:
    return _dt(t).strftime("%Y-%m-%d %I:%M:%S %p %Z").strip()


_FORMATS = ("%Y-%m-%d %I:%M:%S %p", "%Y-%m-%d %I:%M %p", "%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S")


def parse_local(s: str | None) -> float | None:
    """Parse a local wall-clock string (12h with AM/PM, or 24h machine formats); a trailing zone
    abbreviation such as 'PDT' is ignored and the configured zone is used."""
    if not s:
        return None
    s = re.sub(r"\s+(?!AM$|PM$)[A-Z]{2,5}$", "", str(s).strip().strip("[]"))
    for f in _FORMATS:
        try:
            d = datetime.strptime(s, f)
        except ValueError:
            continue
        d = d.replace(tzinfo=_TZ) if _TZ else d.astimezone()
        return d.timestamp()
    return None


def now() -> float:
    return time.time()


def mins(seconds: float) -> int:
    return int(seconds // 60)
