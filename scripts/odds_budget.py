#!/usr/bin/env python3
"""A hard ceiling on Odds API spend, shared by every job that buys a price.

The free tier is 500 credits a MONTH. The scheduled job runs ~25 times a day and
was spending three credits every run — MLB, NFL spreads, CFB spreads — which is
~75/day and ~2,250/month. It ran dry on 2026-08-26, and the failure was silent in
the worst way: the card reported "no plays" because it had no prices, which looks
exactly like it having judged the board and found nothing.

Two mechanisms, because one is not enough:

  THROTTLE estimates. Fetch at most once per SLOW_H when nothing is close to
  kickoff, tightening to FAST_MIN within NEAR_H of one. That matches when prices
  actually move — measured on this repo's own log, only 7% of CFB games 10+ days
  out moved at all over two days, while 86% of NFL games inside three days did.

  CAP guarantees. Count what has actually been spent this calendar month and
  refuse past MONTHLY_CAP. A throttle can be wrong about how many kickoffs fall
  in a window; a counter cannot.

Per-sport budgets are deliberately unequal. CFB opens 2026-08-29 with ~60 games a
week and carries the best-resolving model on the site; MLB's model is refuted,
contributes nothing to the card, and needs the field reference least.
"""
import json
from datetime import datetime, timezone

MONTHLY_CAP = 420          # of 500; the rest is headroom for browser page loads

# sport -> (slow hours, near-kickoff hours, fast minutes)
# Sized so the THROTTLE alone lands under MONTHLY_CAP (~360/month), leaving the
# cap as a backstop rather than something that binds in the last week of the
# month and silently stops capture during the games that matter most.
BUDGET = {
    "mlb":   (24.0, 2.0, 120.0),   # refuted model, contributes nothing to the card
    "nfl":   (8.0,  3.0,  90.0),
    "ncaaf": (8.0,  3.0,  90.0),   # opens 2026-08-29, best-resolving model here
    "wnba":  (24.0, 2.0, 120.0),
}
DEFAULT = (12.0, 2.0, 120.0)


def _now():
    return datetime.now(timezone.utc)


def spend_ok(store, sport, starts=()):
    """True if a credit may be spent on `sport` right now, and record it if so.

    `store` is the job's own JSON dict — the counter rides along with the data it
    protects, so it survives restarts without another file to keep in sync.
    """
    now = _now()
    month = now.strftime("%Y-%m")

    ledger = store.setdefault("oddsBudget", {})
    if ledger.get("month") != month:
        ledger.clear()
        ledger.update({"month": month, "used": 0})
    if ledger.get("used", 0) >= MONTHLY_CAP:
        return False

    slow_h, near_h, fast_min = BUDGET.get(sport, DEFAULT)
    stamps = store.setdefault("oddsFetchedAt", {})
    last = stamps.get(sport)
    try:
        age_min = (now - datetime.fromisoformat(last)).total_seconds() / 60.0 if last else 1e9
    except Exception:
        age_min = 1e9

    soon = False
    for st in starts:
        try:
            t = datetime.fromisoformat((st or "").replace("Z", "+00:00"))
        except Exception:
            continue
        if 0 <= (t - now).total_seconds() / 3600.0 <= near_h:
            soon = True
            break

    if age_min < (fast_min if soon else slow_h * 60):
        return False

    stamps[sport] = now.isoformat(timespec="seconds")
    ledger["used"] = ledger.get("used", 0) + 1
    return True


def summary(store):
    l = store.get("oddsBudget") or {}
    return f"odds credits {l.get('used', 0)}/{MONTHLY_CAP} this {l.get('month', '?')}"
