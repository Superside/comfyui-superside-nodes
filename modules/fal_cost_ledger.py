"""
Running record of every fal.ai call this ComfyUI process has made.

`APIClientMixin.call_api` appends one entry per successful call, so every node
in this package is tracked without needing any per-node change. The Fal Cost
Report node reads the ledger back and formats it.

The ledger lives in memory only - it starts empty on each ComfyUI restart and
nothing is written to disk.
"""

import threading
import time

from . import fal_pricing

# Entries are append-only. Each is a dict:
#   seq, ts, node, endpoint, usd (float|None), detail, note, reason, elapsed_s
_ENTRIES = []
_LOCK = threading.Lock()
_SEQ = 0

# Keep memory bounded on a long-running server. 5000 calls is far more than any
# single session and still trivial in RAM.
MAX_ENTRIES = 5000


def record(node, endpoint, arguments, result, elapsed_s=None):
    """Price one completed fal call and append it to the ledger."""
    global _SEQ

    priced = fal_pricing.estimate(endpoint, arguments, result)
    with _LOCK:
        _SEQ += 1
        entry = {
            "seq": _SEQ,
            "ts": time.time(),
            "node": node,
            "endpoint": endpoint,
            "usd": priced["usd"],
            "detail": priced["detail"],
            "note": priced["note"],
            "reason": priced["reason"],
            "elapsed_s": elapsed_s,
        }
        _ENTRIES.append(entry)
        if len(_ENTRIES) > MAX_ENTRIES:
            del _ENTRIES[: len(_ENTRIES) - MAX_ENTRIES]
    return entry


def entries(after_seq=0):
    """Every entry with seq > after_seq, oldest first."""
    with _LOCK:
        return [dict(e) for e in _ENTRIES if e["seq"] > after_seq]


def last_seq():
    with _LOCK:
        return _SEQ


def clear():
    """Drop every entry (the sequence counter keeps running)."""
    with _LOCK:
        count = len(_ENTRIES)
        _ENTRIES.clear()
    return count


def totals(rows):
    """Aggregate a list of entries."""
    known = [r for r in rows if r["usd"] is not None]
    unknown = [r for r in rows if r["usd"] is None]
    return {
        "calls": len(rows),
        "usd": sum(r["usd"] for r in known),
        "priced_calls": len(known),
        "unpriced_calls": len(unknown),
        "unpriced": unknown,
    }


def _group(rows):
    """Group entries by (node, endpoint), preserving first-seen order."""
    order = []
    groups = {}
    for row in rows:
        key = (row["node"], row["endpoint"])
        if key not in groups:
            groups[key] = {
                "node": row["node"],
                "endpoint": row["endpoint"],
                "calls": 0,
                "usd": 0.0,
                "priced_calls": 0,
                "unpriced_calls": 0,
                "details": [],
                "reason": row["reason"],
                "note": row["note"],
            }
            order.append(key)
        group = groups[key]
        group["calls"] += 1
        if row["usd"] is None:
            group["unpriced_calls"] += 1
        else:
            group["priced_calls"] += 1
            group["usd"] += row["usd"]
        if row["detail"] and row["detail"] not in group["details"]:
            group["details"].append(row["detail"])
    return [groups[k] for k in order]


def format_report(rows, title, currency="USD"):
    """Human-readable cost breakdown for the given entries."""
    width = 72
    lines = [title, "=" * width]

    if not rows:
        lines.append("No fal.ai calls recorded yet.")
        lines.append("")
        lines.append("Prices from fal's catalogue as of {}.".format(fal_pricing.SOURCE_DATE))
        return "\n".join(lines)

    summary = totals(rows)

    for group in _group(rows):
        if group["unpriced_calls"] and not group["priced_calls"]:
            amount = "     n/a"
        else:
            amount = "${:8.4f}".format(group["usd"])
        lines.append("{:>3} x  {:<44} {}".format(group["calls"], group["node"][:44], amount))
        lines.append("       {}".format(group["endpoint"]))
        for detail in group["details"][:3]:
            lines.append("       - {}".format(detail))
        if group["unpriced_calls"]:
            lines.append("       ! {} call(s) not priced".format(group["unpriced_calls"]))

    lines.append("-" * width)
    lines.append("TOTAL  {:>7} priced call(s){:>34}".format(
        summary["priced_calls"], "${:.4f} {}".format(summary["usd"], currency)
    ))

    if summary["unpriced_calls"]:
        lines.append("")
        lines.append("{} call(s) could not be priced:".format(summary["unpriced_calls"]))
        seen = set()
        for row in summary["unpriced"]:
            key = (row["endpoint"], row["reason"])
            if key in seen:
                continue
            seen.add(key)
            lines.append("  - {}".format(row["endpoint"]))
            lines.append("      {}".format(row["reason"]))
        lines.append("")
        lines.append("The total above EXCLUDES those calls.")

    lines.append("")
    lines.append("Estimates from fal's published prices as of {} - fal is the".format(
        fal_pricing.SOURCE_DATE
    ))
    lines.append("only source of truth for what you are actually billed.")
    return "\n".join(lines)
