#!/usr/bin/env python3
"""Two-logit market test for the MLB FULL-GAME model vs Kalshi KXMLBGAME.

The F5 market was measured and refuted; the full-game market was gated on the
family resemblance rather than evidence. This closes that gap.

There is no server-side full-game prediction log, so the model is REPLAYED
point-in-time:
  * games, starters and their K-BB% qualities come from data/mlb_f5_predictions.json
    (the F5 logger already stores kbb_quality(), which is exactly the input
    REAL_MODEL takes)
  * team records come from statsapi standings as of the day BEFORE each game.
    `standings?date=D` INCLUDES games played on D, so querying D would be
    lookahead — verified: 2026-08-01 returns 3,334 team-games, 08-02 returns
    3,364, a full slate apart.
  * the Kalshi price is recovered from the git history of data/kalshi_mlb.json,
    taking the last snapshot at or before first pitch — i.e. the closing price.

Usage:  python scripts/mlb_fullgame_test.py [--devig power]
"""

import argparse, json, os, subprocess, sys, urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import market_edge_test as MET

# ── REAL_MODEL, mirrored from index.html (feature order:
#    pythDiff, winPctDiff, rdDiff, kbbDiff, spKnown, homeAdv) ──
COEF = [0.33448, 0.83292, 0.12567, 1.90351, -0.08084, 0.07588]
INTERCEPT = 0.11408


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def shrink(val, n, prior=0.5, k=20):
    return (val * n + prior * k) / (n + k)


def pythagorean(rs, ra):
    if not rs or not ra or rs <= 0 or ra <= 0:
        return 0.5
    e = 1.83
    return rs ** e / (rs ** e + ra ** e)


def model_prob(h, a, hkbb, akbb):
    """Straight port of realModelRawStats()."""
    hg, ag = h["wins"] + h["losses"], a["wins"] + a["losses"]
    hPs, aPs = shrink(pythagorean(h["rs"], h["ra"]), hg), shrink(pythagorean(a["rs"], a["ra"]), ag)
    hWs = shrink(h["wins"] / hg if hg else 0.5, hg)
    aWs = shrink(a["wins"] / ag if ag else 0.5, ag)
    hRd = (h["rs"] - h["ra"]) / hg if hg else 0.0
    aRd = (a["rs"] - a["ra"]) / ag if ag else 0.0
    sp_known = 1 if (hkbb is not None and akbb is not None) else 0
    kbb_diff = (hkbb - akbb) if sp_known else 0.0
    f = [hPs - aPs, hWs - aWs, hRd - aRd, kbb_diff, sp_known, 1.0]
    z = INTERCEPT + sum(c * x for c, x in zip(COEF, f))
    return float(sigmoid(z))


def api(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def team_abbrs():
    d = api("https://statsapi.mlb.com/api/v1/teams?sportId=1")
    return {t["id"]: t["abbreviation"] for t in d["teams"]}


def standings_asof(date_str, ids):
    """Team records through the END of date_str (so pass game_date - 1 day)."""
    d = api("https://statsapi.mlb.com/api/v1/standings?leagueId=103,104"
            "&season=%s&date=%s" % (date_str[:4], date_str))
    out = {}
    for rec in d.get("records", []):
        for t in rec.get("teamRecords", []):
            ab = ids.get(t["team"]["id"])
            if not ab:
                continue
            out[ab] = {"wins": t["wins"], "losses": t["losses"],
                       "rs": t.get("runsScored") or 0, "ra": t.get("runsAllowed") or 0}
    return out


def kalshi_history():
    """[(unix_ts, {event_ticker: {team_subtitle: mid}})] from the snapshot's git log."""
    log = subprocess.run(["git", "log", "--format=%H %ct", "--", "data/kalshi_mlb.json"],
                         cwd=ROOT, capture_output=True, text=True).stdout.strip().splitlines()
    snaps = []
    for line in log:
        sha, ts = line.split()
        blob = subprocess.run(["git", "show", "%s:data/kalshi_mlb.json" % sha],
                              cwd=ROOT, capture_output=True, text=True).stdout
        try:
            d = json.loads(blob)
        except Exception:
            continue
        ev = {}
        for m in (d.get("markets", {}) or {}).get("KXMLBGAME", []) or []:
            try:
                b = float(m.get("yes_bid_dollars")); a = float(m.get("yes_ask_dollars"))
            except Exception:
                continue
            if not (0 < a < 1) or not (0 <= b < 1) or b > a:
                continue
            ev.setdefault(m["event_ticker"], {})[m.get("yes_sub_title")] = (b + a) / 2.0
        snaps.append((int(ts), ev))
    snaps.sort(key=lambda x: x[0])
    return snaps


def finals_for(dates):
    """gamePk -> 1 if home won."""
    out = {}
    for ymd in sorted(dates):
        d = api("https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=%s" % ymd)
        for day in d.get("dates", []):
            for g in day.get("games", []):
                if g.get("status", {}).get("abstractGameState") != "Final":
                    continue
                t = g["teams"]
                if t["home"].get("score") is None:
                    continue
                out[g["gamePk"]] = 1 if t["home"]["score"] > t["away"]["score"] else 0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devig", default="power",
                    choices=["power", "multiplicative", "additive", "shin"])
    ap.add_argument("--out", default=os.path.join(ROOT, "..", "mlb_fullgame_results.txt"))
    a = ap.parse_args()

    E = json.load(open(os.path.join(ROOT, "data", "mlb_f5_predictions.json"),
                       encoding="utf-8"))["entries"]
    ids = team_abbrs()
    snaps = kalshi_history()
    print("kalshi snapshots in git history: %d" % len(snaps), flush=True)

    # point-in-time standings, one call per distinct prior day
    days = sorted({e["date"] for e in E})
    stand = {}
    for d in days:
        prior = (datetime.strptime(d, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
        stand[d] = standings_asof(prior, ids)
    finals = finals_for(days)
    print("standings pulled for %d days, finals for %d games" % (len(stand), len(finals)), flush=True)

    Pm, Pk, y, skipped = [], [], [], {"noticker": 0, "nosnap": 0, "nostand": 0, "nofinal": 0}
    for e in E:
        gp = e["gamePk"]
        if gp not in finals:
            skipped["nofinal"] += 1; continue
        st = stand.get(e["date"]) or {}
        h, aw = st.get(e["home"]), st.get(e["away"])
        if not h or not aw:
            skipped["nostand"] += 1; continue
        tick = ((e.get("kalshi") or {}).get("ticker") or "")
        if not tick.startswith("KXMLBF5-"):
            skipped["noticker"] += 1; continue
        game_tick = "KXMLBGAME-" + tick.split("-", 1)[1]
        start = int(datetime.strptime(e["gameDate"], "%Y-%m-%dT%H:%M:%SZ")
                    .replace(tzinfo=timezone.utc).timestamp())
        # last snapshot at or before first pitch = the closing price
        price = None
        for ts, ev in snaps:
            if ts > start:
                break
            if game_tick in ev and len(ev[game_tick]) == 2:
                price = ev[game_tick]
        if not price:
            skipped["nosnap"] += 1; continue
        # which subtitle is the home team? Kalshi uses city names; the ticker
        # ends <AWAY><HOME> in statsapi abbreviations, so match on order.
        subs = list(price.items())
        # event ticker tail is e.g. "26AUG121340BALMIN" -> away BAL, home MIN
        tail = game_tick.split("-", 1)[1]
        home_ab = e["home"]
        # pick the subtitle whose first letters match the home abbreviation's city
        home_p = None
        for sub, p in subs:
            if sub and sub[:3].upper() == home_ab[:3].upper():
                home_p = p
        if home_p is None:
            # fall back: Kalshi lists markets in <away>,<home> order within an event
            home_p = subs[-1][1]
        away_p = [p for s, p in subs if p != home_p]
        away_p = away_p[0] if away_p else (1 - home_p)
        dv = MET.devig([home_p, away_p], a.devig)
        if dv is None:
            skipped["nosnap"] += 1; continue
        mp = model_prob(h, aw, (e.get("starters") or {}).get("homeKBB"),
                        (e.get("starters") or {}).get("awayKBB"))
        Pm.append([mp, 1 - mp]); Pk.append(dv)
        y.append(0 if finals[gp] else 1)

    Pm, Pk, y = np.array(Pm), np.array(Pk), np.array(y)
    L = ["MLB FULL-GAME MODEL vs KALSHI KXMLBGAME  (de-vig = %s)" % a.devig,
         "usable games: %d   skipped: %s" % (len(y), skipped)]
    if len(y) >= 30:
        L.append(MET.report("MLB full game  vs  Kalshi closing snapshot",
                            MET.fit_pool(Pm, Pk, y)))
        acc = float(((Pm[:, 0] >= 0.5) == (y == 0)).mean())
        mkt = float(((Pk[:, 0] >= 0.5) == (y == 0)).mean())
        L.append("  straight-up accuracy: model %.3f | market %.3f | home base %.3f"
                 % (acc, mkt, float((y == 0).mean())))
    else:
        L.append("  too few usable games to test.")
    text = "\n".join(L)
    open(a.out, "w", encoding="utf-8").write(text)
    sys.stdout.write(text.encode("ascii", "replace").decode("ascii") + "\n")


if __name__ == "__main__":
    main()
