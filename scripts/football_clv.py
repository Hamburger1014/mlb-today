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
