#!/usr/bin/env python3
"""Football CLV priced Walters-style: what was the NUMBER worth, not the price?

The football views are market-only, so there is no model to grade. What there is
to grade is the number. If you took a side when the line opened, the closing line
says what the market eventually thought, and the gap between the two is the only
honest read on whether taking it early was right.

Every CLV number on this site so far has been in price terms, which is the wrong
unit for a spread. Moving from -3 to -2.5 barely changes the price and changes
the bet enormously, because 15.39% of NFL games land on exactly 3. This grades
the open against the close through walters.number_edge, so a half point at 3 is
worth roughly four times a half point at 5 — as it should be.

Reads data/football_lines.json. Writes nothing; this is a measurement.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import walters as W

OUT = os.path.join(os.path.dirname(HERE), "data", "football_lines.json")


def main():
    games = json.load(open(OUT))["games"]
    for league, sigma in (("NFL", W.NFL_SIGMA), ("CFB", W.CFB_SIGMA)):
        rows = []
        for g in games.values():
            if g.get("league") != league:
                continue
            o, c = g.get("open"), g.get("closing")
            if not o or not c:
                continue
            if o.get("spHomeLine") is None or c.get("spHomeLine") is None:
                continue
            # Grade the OPENING number against the CLOSING one: the close is the
            # market's final word, so it plays the role the field plays elsewhere.
            e = W.number_edge(o["spHomeLine"], o.get("spHomeOdds") or -110,
                              o.get("spAwayOdds") or -110,
                              c["spHomeLine"], c.get("spHomeOdds") or -110,
                              c.get("spAwayOdds") or -110, sigma)
            if e["home"] is None:
                continue
            # Half the vig is the baseline: a bet struck at exactly the closing
            # number still pays the hold, so that is what "no edge" looks like.
            best = max(e["home"], e["away"])
            side = "home" if e["home"] >= e["away"] else "away"
            rows.append((best, side, g, o["spHomeLine"], c["spHomeLine"], e["home"]))

        if not rows:
            print(f"{league}: nothing gradable yet")
            continue
        rows.sort(key=lambda r: r[0], reverse=True)   # ties would compare dicts
        big = [r for r in rows if r[0] >= 0.02]
        hindsight = sum(r[0] for r in rows) / len(rows)
        # THE CONTROL, and the only line here that means anything. Taking the
        # better side is chosen AFTER seeing which way the line moved, so it beats
        # the close by construction — the same selection trap that made a raw
        # +0.519pp CLV look decisive on the MLB log until a control showed +0.493
        # of it was just backing favourites. Committing to the home side in
        # advance is a rule you could actually have followed.
        blind = sum(r[5] for r in rows) / len(rows)
        print(f"\n{league}  n={len(rows)}")
        print(f"  HINDSIGHT (picks the side that won) : {hindsight*100:+.2f}pp  <- not a strategy")
        print(f"  CONTROL   (always home, decided up front) : {blind*100:+.2f}pp")
        print(f"  vig-only baseline, no edge at all         : -2.38pp")
        print(f"  hindsight bets clearing a 2pp bar   : {len(big)} ({100*len(big)/len(rows):.0f}%)")
        print("  most number-value that was on the table (hindsight):")
        for best, side, g, ol, cl, _h in rows[:6]:
            t = g["home"] if side == "home" else g["away"]
            print(f"     {g['away']} @ {g['home']:<4} take {t:<4} at {ol:+.1f} "
                  f"(closed {cl:+.1f})  {best*100:+5.2f}pp  {W.walters_units(best)}u")


if __name__ == "__main__":
    main()


# ── PAPER PICK CLV ────────────────────────────────────────────────────────
# The section above grades the NUMBER (open vs close) with no model involved.
# This grades the MODEL's paper picks, recorded by football_log.py at two fixed
# horizons, against the number the market closed at.
#
# CLV IS IN POINTS, NOT PRICE. Moving 3 -> 2.5 barely moves the price and changes
# the bet enormously. Sign convention, pinned by cases below:
#
#     pts = lineTaken - lineAtClose        (on the side actually taken)
#
# Took home -3.5, closed -4.5  ->  -3.5 - -4.5 = +1.0   you got the better number
# Took home -4.5, closed -3.5  ->  -4.5 - -3.5 = -1.0   you laid more than you had to
# Took away +3.5, closed +4.5  ->  +3.5 - +4.5 = -1.0   you took fewer points
#
# THE CONTROL IS NOT OPTIONAL. Measured on MLB, simply backing the market's
# favourite earned +0.493pp of raw CLV, and a model that mostly picked favourites
# collected that drift and looked like it had +0.519pp of skill. The incremental
# — model CLV minus the control on the SAME games — was +0.026pp, i.e. nothing.
# Report the paired difference; the raw number is context, not evidence.


def _side_line(g, side, block):
    """That side's line from a snapshot block, or None."""
    h = (g.get(block) or {}).get("spHomeLine")
    if h is None:
        return None
    return h if side == "home" else -h


def pick_clv():
    games = json.load(open(OUT))["games"]
    for league in ("CFB", "NFL"):
        for tag in ("early", "late"):
            rows = []
            for g in games.values():
                if g.get("league") != league:
                    continue
                p = (g.get("picks") or {}).get(tag)
                if not p:
                    continue
                close = _side_line(g, p["side"], "closing")
                if close is None:
                    continue
                # A pick is only measurable once the line has stopped moving, i.e.
                # once `closing` is genuinely the close. Before kickoff it is still
                # being overwritten, so grading it now measures nothing.
                if (g.get("closing") or {}).get("at", "") <= p["at"]:
                    continue
                ctrl_taken = _side_line(g, p["ctrlSide"], "closing")
                ctrl_at_pick = p["homeLine"] if p["ctrlSide"] == "home" else -p["homeLine"]
                rows.append({
                    "model": p["line"] - close,
                    "ctrl": ctrl_at_pick - ctrl_taken,
                    "edge": p["edgePts"],
                })
            if not rows:
                print(f"{league} {tag}: no measurable picks yet "
                      f"(a pick becomes measurable once the line has closed)")
                continue
            n = len(rows)
            m = sum(r["model"] for r in rows) / n
            c = sum(r["ctrl"] for r in rows) / n
            d = [r["model"] - r["ctrl"] for r in rows]
            inc = sum(d) / n
            sd = (sum((x - inc) ** 2 for x in d) / (n - 1)) ** 0.5 if n > 1 else float("nan")
            se = sd / (n ** 0.5) if n > 1 else float("nan")
            beat = sum(1 for r in rows if r["model"] > r["ctrl"])
            print(f"{league} {tag}: n={n}  model {m:+.3f} pts | control {c:+.3f} pts | "
                  f"INCREMENTAL {inc:+.3f} +/- {se:.3f} (z={inc/se:+.2f})  beat control {beat}/{n}"
                  if se == se and se > 0 else
                  f"{league} {tag}: n={n}  model {m:+.3f} | control {c:+.3f} | incremental {inc:+.3f}")


if __name__ == "__main__":
    print()
    pick_clv()
