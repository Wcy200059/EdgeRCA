"""Small deterministic statistics shared by the fixed CPU tools."""

from __future__ import annotations

import hashlib
import math
from statistics import fmean, median


def number(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError, OverflowError):
        return None


def identifier(*parts):
    return hashlib.sha256(repr(parts).encode()).hexdigest()[:20]


def unique(values):
    return list(dict.fromkeys(values))


def distribution(values):
    """Exact statistics; quantiles use nearest rank, including small samples."""
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "mean": None, "median": None, "p50": None,
                "p95": None, "min": None, "max": None}
    return {"count": len(ordered), "mean": fmean(ordered), "median": median(ordered),
            "p50": ordered[math.ceil(len(ordered) * .5) - 1],
            "p95": ordered[math.ceil(len(ordered) * .95) - 1],
            "min": ordered[0], "max": ordered[-1]}


def change(before, after):
    if before is None or after is None:
        return {"direction": "unknown", "delta": None, "ratio": None}
    return {"direction": "up" if after > before else "down" if after < before else "unchanged",
            "delta": after - before, "ratio": after / before if before != 0 else None}


def period(start, end):
    return {"start_ms": start, "end_ms": end, "seconds": (end - start) / 1000,
            "bounds": "[start,end)"}
