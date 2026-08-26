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
import urllib.request
from datetime import datetime, timezone

# FALLBACK ONLY. The real ceiling is whatever the API says is left — every
# successful response carries x-requests-remaining, so the budget calibrates
# itself to the active tier instead of trusting a number someone hand-edited
# after an upgrade and might forget to change back. This applies when no
# response has been seen yet this month.
MONTHLY_CAP = 420
RESERVE = 60               # never spend the last of the tank; leave it for the page
PROBE_H = 3.0              # how stale an "empty" reading gets before we re-check

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

    # Prefer what the API last told us over any local guess.
    rem = ledger.get("remaining")
    if rem is not None:
        if rem <= RESERVE:
            # DEADLOCK GUARD. Refusing on an empty reading means never fetching,
            # and never fetching means never learning the tank refilled. A month
            # rollover clears the ledger so the monthly reset recovers on its
            # own — but a mid-month UPGRADE does not, and that is precisely what
            # happened on 2026-08-26: the tier went to 20,000 credits while this
            # sat at remaining 0 and would have stayed blind indefinitely. Let a
            # stale "empty" spend one credit to re-check.
            try:
                age_h = (now - datetime.fromisoformat(ledger["checkedAt"])).total_seconds() / 3600.0
            except Exception:
                age_h = 1e9
            if age_h < PROBE_H:
                return False
            stamps = store.setdefault("oddsFetchedAt", {})
            stamps[sport] = now.isoformat(timespec="seconds")
            ledger["used"] = ledger.get("used", 0) + 1
            return True     # a probe: one call to find out whether it refilled
    elif ledger.get("used", 0) >= MONTHLY_CAP:
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
    rem = l.get("remaining")
    tank = f"{rem} left" if rem is not None else f"cap {MONTHLY_CAP}"
    return f"odds: {l.get('used', 0)} spent this {l.get('month', '?')}, {tank}"


def note_response(store, remaining=None, exhausted=False):
    """Record what the API reported about the tank.

    `remaining` comes from x-requests-remaining on a successful call; `exhausted`
    is set when the API answers OUT_OF_USAGE_CREDITS, which is the same fact
    stated as an error. Either way the next spend_ok() uses the truth rather than
    a hardcoded ceiling.
    """
    ledger = store.setdefault("oddsBudget", {})
    if exhausted:
        ledger["remaining"] = 0
    elif remaining is not None:
        try:
            ledger["remaining"] = int(float(remaining))
        except (TypeError, ValueError):
            pass
    ledger["checkedAt"] = _now().isoformat(timespec="seconds")


def fetch_json(url, store, timeout=30, tries=3):
    """GET JSON through the Worker and record the quota it reports.

    Returns (data or None). Callers keep their own error handling; this exists so
    the credit accounting cannot be forgotten at a call site.
    """
    last = None
    for _ in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0",
                                                       "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                note_response(store, remaining=r.headers.get("x-requests-remaining"))
                return json.load(r)
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", "replace")[:200]
            except Exception:
                pass
            if e.code == 401 and "OUT_OF_USAGE_CREDITS" in body:
                note_response(store, exhausted=True)
                raise RuntimeError("odds api out of credits") from e
            note_response(store, remaining=e.headers.get("x-requests-remaining"))
            last = e
        except Exception as e:
            last = e
    raise last
