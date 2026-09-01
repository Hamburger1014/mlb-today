#!/usr/bin/env python3
"""Does the MLB full-game model have ANY skill? Not versus the market — versus
a constant.

mlb_fullgame_test.py answered the market question on 134 games (it is capped by
how far the Kalshi snapshots go back) and found nothing. This asks the weaker,
more basic question on the whole season, where there is far more power:

    can the model beat "the home team wins 52.3% of the time"?

WHY THIS IS A SEPARATE TEST. A model can fail against a sharp market and still
be useful — the market is a hard opponent. Failing against a CONSTANT is a
different and much worse result: it means the features carry no game-level
signal at all, and no amount of shrinking toward the price will rescue it.

METHOD. Walk-forward: for each slate, fit on every PRIOR game only and predict
that day. Every game after the burn-in is out of sample, which is why this gets
1,369 test games where the market test gets 134. Significance is a paired
bootstrap on per-game log-loss, because accuracy alone throws away the
confidence attached to each pick.

Point-in-time inputs, same as mlb_fullgame_test:
  * standings as of the day BEFORE the game (statsapi `standings?date=D`
    INCLUDES games played on D, so querying D would be lookahead)
  * each starter's K-BB% from this season's starts strictly BEFORE the game,
    requiring >=100 batters faced

Usage:  python scripts/mlb_skill_test.py [--cache PATH] [--burn 300]
"""
import argparse, importlib.util, json, os, sys, urllib.request
from datetime import datetime, timedelta

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

_ms = importlib.util.spec_from_file_location("mt", os.path.join(HERE, "mlb_fullgame_test.py"))
MT = importlib.util.module_from_spec(_ms); _ms.loader.exec_module(MT)
_cs = importlib.util.spec_from_file_location("cal", os.path.join(HERE, "calibration.py"))
C = importlib.util.module_from_spec(_cs); _cs.loader.exec_module(C)

# The shipped coefficients, mirrored from index.html realModelRawStats().
FEATS = ["pyth", "win", "rd", "kbb", "sp"]
SHIP = np.array(MT.COEF[:5] + [MT.INTERCEPT])
START, END = "2026-04-01", "2026-08-13"


def api(u):
    r = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
    return json.load(urllib.request.urlopen(r, timeout=30))


def build_rows(cache):
    """Replay the season, keeping the FEATURES and not just the probability —
    without them the model can only be scored, never decomposed."""
    if cache and os.path.exists(cache):
        return json.load(open(cache))
    ids = MT.team_abbrs()
    sched = api("https://statsapi.mlb.com/api/v1/schedule?sportId=1"
                "&startDate=%s&endDate=%s&hydrate=probablePitcher" % (START, END))
    games, pids = [], set()
    for day in sched.get("dates", []):
        for g in day.get("games", []):
            if g.get("status", {}).get("abstractGameState") != "Final":
                continue
            t = g.get("teams", {}); h, a = t.get("home", {}), t.get("away", {})
            if h.get("score") is None or a.get("score") is None or h["score"] == a["score"]:
                continue                      # ties carry no label
            hp = (h.get("probablePitcher") or {}).get("id")
            ap = (a.get("probablePitcher") or {}).get("id")
            games.append({"date": day["date"], "home": ids.get(h.get("team", {}).get("id")),
                          "away": ids.get(a.get("team", {}).get("id")),
                          "hs": h["score"], "as": a["score"], "hp": hp, "ap": ap})
            for x in (hp, ap):
                if x: pids.add(x)
    games = [g for g in games if g["home"] and g["away"]]
    print("final games %d, distinct starters %d" % (len(games), len(pids)), flush=True)

    logs = {}
    for i, pid in enumerate(sorted(pids)):
        try:
            gl = api("https://statsapi.mlb.com/api/v1/people/%d/stats"
                     "?stats=gameLog&group=pitching&season=2026" % pid)
            sp = (gl.get("stats") or [{}])[0].get("splits") or []
            logs[pid] = [{"d": s.get("date"), "k": s["stat"].get("strikeOuts") or 0,
                          "bb": s["stat"].get("baseOnBalls") or 0,
                          "bf": s["stat"].get("battersFaced") or 0}
                         for s in sp if s.get("date")]
        except Exception:
            logs[pid] = []
        if i % 75 == 0:
            print("  pitcher %d/%d" % (i, len(pids)), flush=True)

    # These MUST match index.html's _kbbQuality or the test measures a model that
    # is not the one shipping. They did not until 2026-08-31: this used a hard
    # `bf >= 100 else None`, which is neither the floor nor the ramp the page
    # applies, and it silently withheld a starter quality on 403 of 1,674 games
    # that the live model rates perfectly well.
    KBB_LG, KBB_WBF, BF_FLOOR = 0.14138, 250, 40

    def kbb_asof(pid, date):
        """Starter K-BB% regressed toward league average by sample size, exactly
        as index.html does it: null below a 40-BF floor, then a linear ramp to
        full weight at 250 batters faced."""
        if not pid:
            return None
        k = bb = bf = 0
        for e in logs.get(pid, []):
            if e["d"] < date:                 # strictly before: no lookahead
                k += e["k"]; bb += e["bb"]; bf += e["bf"]
        if bf < BF_FLOOR:
            return None
        wc = min(1.0, bf / KBB_WBF)
        return wc * ((k - bb) / bf) + (1 - wc) * KBB_LG

    days = sorted({g["date"] for g in games})
    stand = {}
    for i, d in enumerate(days):
        prior = (datetime.strptime(d, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            stand[d] = MT.standings_asof(prior, ids)
        except Exception:
            stand[d] = {}
        if i % 40 == 0:
            print("  standings %d/%d" % (i, len(days)), flush=True)

    rows = []
    for g in games:
        st = stand.get(g["date"]) or {}
        h, a = st.get(g["home"]), st.get(g["away"])
        if not h or not a:
            continue
        hg, ag = h["wins"] + h["losses"], a["wins"] + a["losses"]
        if hg < 10 or ag < 10:                # April records are noise
            continue
        hk, ak = kbb_asof(g["hp"], g["date"]), kbb_asof(g["ap"], g["date"])
        spk = 1 if (hk is not None and ak is not None) else 0
        rows.append({
            "date": g["date"], "home": g["home"], "away": g["away"],
            "y": 1 if g["hs"] > g["as"] else 0,
            "pyth": MT.shrink(MT.pythagorean(h["rs"], h["ra"]), hg)
                    - MT.shrink(MT.pythagorean(a["rs"], a["ra"]), ag),
            "win": MT.shrink(h["wins"] / hg, hg) - MT.shrink(a["wins"] / ag, ag),
            "rd": (h["rs"] - h["ra"]) / hg - (a["rs"] - a["ra"]) / ag,
            "kbb": (hk - ak) if spk else 0.0, "sp": spk,
        })
    if cache:
        json.dump(rows, open(cache, "w"))
    return rows


def fit(rs, feats, l2):
    """Newton-fit a logit. The intercept is deliberately NOT regularised: the
    base home-field rate is a real quantity we know well, and shrinking it
    toward zero would force the other features to re-absorb it. That exact
    mistake compressed the football ratings (see cfb_model.py, RIDGE)."""
    X = np.column_stack([[r[f] for r in rs] for f in feats] + [np.ones(len(rs))])
    y = np.array([r["y"] for r in rs], float)
    b = np.zeros(X.shape[1])
    R = l2 * np.eye(X.shape[1]); R[-1, -1] = 0.0
    for _ in range(80):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
        W = p * (1 - p) + 1e-9
        b += np.linalg.solve(X.T @ (X * W[:, None]) + R, X.T @ (y - p) - R @ b)
    return b


def predict(rs, feats, b):
    X = np.column_stack([[r[f] for r in rs] for f in feats] + [np.ones(len(rs))])
    return 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))


def walk_forward(rows, burn, l2):
    """Fit on everything before each slate, predict that slate. Returns the
    shipped model, a daily refit, a pythagorean-only fit, and the running base
    rate, all on the same games."""
    days = sorted({r["date"] for r in rows})
    out = {k: [] for k in ("ship", "refit", "pyth", "base")}
    y = []
    for d in days:
        prior = [r for r in rows if r["date"] < d]
        today = [r for r in rows if r["date"] == d]
        if len(prior) < burn or not today:
            continue
        bf, bp = fit(prior, FEATS, l2), fit(prior, ["pyth"], l2)
        base = float(np.mean([r["y"] for r in prior]))
        out["ship"].extend(predict(today, FEATS, SHIP))
        out["refit"].extend(predict(today, FEATS, bf))
        out["pyth"].extend(predict(today, ["pyth"], bp))
        out["base"].extend([base] * len(today))
        y.extend(r["y"] for r in today)
    return {k: np.clip(np.array(v, float), 1e-6, 1 - 1e-6) for k, v in out.items()}, \
           np.array(y, float)


def logloss(p, y):
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(ROOT, "analysis", "mlb_features.json"))
    ap.add_argument("--burn", type=int, default=300)
    ap.add_argument("--l2", type=float, default=1.0)
    a = ap.parse_args()
    if a.cache:
        os.makedirs(os.path.dirname(a.cache), exist_ok=True)

    rows = build_rows(a.cache)
    rows.sort(key=lambda r: r["date"])
    y_all = np.array([r["y"] for r in rows], float)
    print("\nreplayed %d games; home win rate %.4f" % (len(rows), y_all.mean()))

    P, y = walk_forward(rows, a.burn, a.l2)
    print("walk-forward test games: %d (burn-in %d)\n" % (len(y), a.burn))

    print("  %-28s %7s  %7s  %8s  %7s  %7s" % ("", "acc", "Brier", "logloss", "res", "rel"))
    for k, lab in (("base", "always home (prior rate)"), ("ship", "SHIPPED coefficients"),
                   ("refit", "refit walk-forward"), ("pyth", "pythagorean only")):
        p = P[k]; d = C.brier_decomp(list(p), list(y))
        print("  %-28s %6.2f%%  %.4f  %8.4f  %.4f  %.4f"
              % (lab, 100 * ((p >= .5) == (y == 1)).mean(), ((p - y) ** 2).mean(),
                 logloss(p, y).mean(), d["resolution"], d["reliability"]))

    # THE test: the model against a constant, with a confidence interval.
    rng = np.random.default_rng(11)
    dif = logloss(P["base"], y) - logloss(P["ship"], y)   # >0 = model better
    bs = dif[rng.integers(0, len(dif), (20000, len(dif)))].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    acc, base = ((P["ship"] >= .5) == (y == 1)).mean(), y.mean()
    se = (0.25 / len(y)) ** 0.5
    print("\nSHIPPED vs ALWAYS-HOME on %d games" % len(y))
    print("  log-loss gain %+.5f/game   95%% CI [%+.5f, %+.5f]   P(model better) %.3f"
          % (dif.mean(), lo, hi, (bs > 0).mean()))
    print("  accuracy %.2f%% vs %.2f%% base = %+.2fpp, ~%.2f SE"
          % (100 * acc, 100 * base, 100 * (acc - base), (acc - base) / se))
    print("\n  VERDICT: %s"
          % ("model beats the constant at 95%" if lo > 0 else
             "NO demonstrable skill - the interval spans zero"))


if __name__ == "__main__":
    main()
