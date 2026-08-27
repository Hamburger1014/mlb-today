#!/usr/bin/env python3
"""Does the WNBA model beat a CONSTANT? Fourth and last of the skill tests, and
the one that matters most: WNBA is the only sport carrying betting weight
(WNBASP w=0.40, WNBAML w=0.15), so this is the model with money behind it.

Companions: analysis/mlb_no_skill.md (no skill), nfl_skill.md (skill, scale bug),
cfb_skill.md (strongest, clean).

TWO SOURCES OF LOOKAHEAD HAD TO BE REMOVED, and both would have flattered the
model badly:

 1. THE SHIPPED COEFFICIENTS ALREADY SAW THE TEST GAMES. wnba_log.FIT is
    analysis/wnba_fit_v2.json, trained on 1,246 games INCLUDING 2026. Scoring
    the shipped fit on 2026 is in-sample. So the margin coefficients are refit
    WALK-FORWARD here: at each slate, least squares on prior games only. The
    shipped fit is still reported, clearly marked, for contrast.

 2. `priors2025` SEEDS EVERY TEAM WITH 2025 RATINGS regardless of date. Live
    that is fine — it is last season's prior, worth priorW0=4 games and halved
    after 25. In a 2022-2024 replay it is the future. Ratings here are seeded at
    league average instead, and the test burns in two full seasons so the seed
    has decayed to nothing either way.

WHAT THIS TEST CANNOT DO. There is no historical point-in-time injury feed, so
the injMiss term is dropped (miss=0 for both sides). market_test_results records
that the injury feature is where most of the WNBA signal appeared -- and that
its BACKTEST version was lookahead, reading the game's own boxscore. So this
measures the model WITHOUT its most valuable and most dangerous input. Read the
result as a floor on the live model, not as an estimate of it.

Ratings replicate wnba_log.load_ratings offline from analysis/wnba_games.json
(seasonType 2 and 3 only -- preseason is excluded there and here).

Usage:  python scripts/wnba_skill_test.py [--from 2024]
"""
import argparse, importlib.util, json, math, os, sys
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_ws = importlib.util.spec_from_file_location("wl", os.path.join(HERE, "wnba_log.py"))
WL = importlib.util.module_from_spec(_ws); _ws.loader.exec_module(WL)
_cal = importlib.util.spec_from_file_location("cal", os.path.join(HERE, "calibration.py"))
C = importlib.util.module_from_spec(_cal); _cal.loader.exec_module(C)

FIT = WL.FIT
GAMES = os.path.join(ROOT, "analysis", "wnba_games.json")


def load_games():
    """Regular season and playoffs only, chronological. seasonType 1 is
    preseason -- wnba_log.load_ratings drops it and so must this."""
    raw = json.load(open(GAMES))
    out = []
    for g in raw:
        if g.get("seasonType") not in (2, 3):
            continue
        h, a = g.get("home") or {}, g.get("away") or {}
        if h.get("score") is None or a.get("score") is None:
            continue
        if h["score"] == a["score"]:
            continue                       # no ties in basketball, but be safe
        out.append({"date": g["date"], "season": g["seasonYear"],
                    "home": h["abbr"], "away": a["abbr"],
                    "hs": float(h["score"]), "as": float(a["score"])})
    out.sort(key=lambda x: x["date"])
    return out


def ratings_asof(games, upto_idx):
    """Exponentially weighted PF/PA over games[:upto_idx], league-average seeded.

    Mirrors load_ratings' recursion (wpf = wpf*lam + my) but seeds at lgFallback
    rather than priors2025, which would be next-season information in a replay.
    """
    lam = 0.5 ** (1.0 / FIT["halfLife"])
    w0 = float(FIT["priorW0"]); lgf = FIT["lgFallback"]
    st = {t: {"pf": lgf * w0, "pa": lgf * w0, "n": w0, "gp": 0}
          for t in WL.TEAMS}
    for g in games[:upto_idx]:
        for side, my, op in ((g["home"], g["hs"], g["as"]), (g["away"], g["as"], g["hs"])):
            s = st.get(side)
            if s is None:
                continue
            s["pf"] = s["pf"] * lam + my
            s["pa"] = s["pa"] * lam + op
            s["n"] = s["n"] * lam + 1
            s["gp"] += 1
    stats = {t: {"pf": s["pf"] / s["n"], "pa": s["pa"] / s["n"], "wN": s["n"], "gp": s["gp"]}
             for t, s in st.items()}
    played = [s for s in stats.values() if s["gp"] > 0]
    lg = sum(s["pf"] for s in played) / len(played) if played else lgf
    return stats, lg


def eff_pair(stats, lg, home, away):
    """The model's two expected-points terms, straight from wnba_log.predict."""
    h, a = stats.get(home), stats.get(away)
    if not h or not a:
        return None
    hpf = WL.shrink(h["pf"], h["wN"], lg); hpa = WL.shrink(h["pa"], h["wN"], lg)
    apf = WL.shrink(a["pf"], a["wN"], lg); apa = WL.shrink(a["pa"], a["wN"], lg)
    return hpf * apa / lg, apf * hpa / lg


def b2b_map(games):
    """Which teams played the previous calendar day, per game index."""
    last = {}
    out = []
    for g in games:
        d = datetime.strptime(g["date"][:10], "%Y-%m-%d")
        flags = {}
        for side in (g["home"], g["away"]):
            prev = last.get(side)
            flags[side] = 1 if prev is not None and (d - prev).days == 1 else 0
        out.append(flags)
        for side in (g["home"], g["away"]):
            last[side] = d
    return out


def fit_margin(rows):
    """Least squares for mu = eff*(eh0-ea0) + homeEdge + b2b*(bh-ba).
    Same functional form as FIT['margin'], minus injMiss."""
    X = np.array([[r["ediff"], 1.0, r["bdiff"]] for r in rows])
    y = np.array([r["margin"] for r in rows])
    return np.linalg.lstsq(X, y, rcond=None)[0]      # eff, homeEdge, b2b


def fit_k(mus, ys, lo=0.02, hi=0.60, iters=80):
    """winProbK: p = sigmoid(K*mu). Golden section on log loss."""
    m = np.asarray(mus, float); yy = np.asarray(ys, float)
    def loss(k):
        p = np.clip(1 / (1 + np.exp(-np.clip(k * m, -30, 30))), 1e-12, 1 - 1e-12)
        return float(-(yy * np.log(p) + (1 - yy) * np.log(1 - p)).mean())
    gr = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - gr * (b - a), a + gr * (b - a)
    for _ in range(iters):
        if loss(c) < loss(d): b = d
        else: a = c
        c, d = b - gr * (b - a), a + gr * (b - a)
    return (a + b) / 2


def logloss(p, y):
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="test_from", type=int, default=2024)
    a = ap.parse_args()

    games = load_games()
    b2b = b2b_map(games)
    print("loaded %d regular+playoff games, %d-%d"
          % (len(games), min(g["season"] for g in games), max(g["season"] for g in games)))

    # Precompute each game's features against ratings built from everything before it.
    feats = []
    for i, g in enumerate(games):
        stats, lg = ratings_asof(games, i)
        ep = eff_pair(stats, lg, g["home"], g["away"])
        if ep is None:
            feats.append(None); continue
        eh0, ea0 = ep
        feats.append({"ediff": eh0 - ea0, "bdiff": b2b[i][g["home"]] - b2b[i][g["away"]],
                      "margin": g["hs"] - g["as"], "y": 1 if g["hs"] > g["as"] else 0,
                      "season": g["season"], "date": g["date"]})

    idx = [i for i, f in enumerate(feats) if f and f["season"] >= a.test_from]
    print("test games: %d (seasons %d+), burn-in %d games\n"
          % (len(idx), a.test_from, idx[0] if idx else 0))

    P = {k: [] for k in ("base", "homeedge", "wf", "shipped")}
    Y = []
    sig = lambda z: 1 / (1 + math.exp(-max(-30, min(30, z))))
    last_season = None
    for i in idx:
        f = feats[i]
        prior = [feats[j] for j in range(i) if feats[j]]
        if len(prior) < 200:
            continue
        # Refit once per season; within a season the coefficients are stable and
        # refitting per game would cost ~1300 least-squares fits for no gain.
        if f["season"] != last_season:
            coef = fit_margin(prior)
            mus_p = [coef[0] * r["ediff"] + coef[1] + coef[2] * r["bdiff"] for r in prior]
            K = fit_k(mus_p, [r["y"] for r in prior])
            base = float(np.mean([r["y"] for r in prior]))
            he_only = sig(K * coef[1])
            last_season = f["season"]
        mu_wf = coef[0] * f["ediff"] + coef[1] + coef[2] * f["bdiff"]
        mu_sh = (FIT["margin"]["eff"] * f["ediff"] + FIT["margin"]["homeEdge"]
                 + FIT["margin"]["b2b"] * f["bdiff"])
        P["base"].append(base)
        P["homeedge"].append(he_only)
        P["wf"].append(sig(K * mu_wf))
        P["shipped"].append(WL.clamp(sig(FIT["winProbK"] * mu_sh), 0.03, 0.97))
        Y.append(f["y"])

    Y = np.array(Y, float)
    P = {k: np.clip(np.array(v, float), 1e-6, 1 - 1e-6) for k, v in P.items()}
    print("scored %d games   home rate %.4f\n" % (len(Y), Y.mean()))
    print("  %-38s %7s  %7s  %8s  %7s  %7s" % ("", "acc", "Brier", "logloss", "res", "rel"))
    for k, lab in (("base", "always home (prior rate)"),
                   ("homeedge", "home-edge only"),
                   ("wf", "MODEL, coefficients refit walk-forward"),
                   ("shipped", "MODEL, shipped FIT (IN-SAMPLE, inflated)")):
        p = P[k]; d = C.brier_decomp(list(p), list(Y))
        print("  %-38s %6.2f%%  %.4f  %8.4f  %.4f  %.4f"
              % (lab, 100 * ((p >= .5) == (Y == 1)).mean(), ((p - Y) ** 2).mean(),
                 logloss(p, Y).mean(), d["resolution"], d["reliability"]))

    rng = np.random.default_rng(11)
    print("\nPAIRED BOOTSTRAP on per-game log-loss (20k resamples)")
    for k, lab in (("wf", "MODEL(walk-forward) vs always-home"),
                   ("homeedge", "home-edge only vs always-home")):
        dif = logloss(P["base"], Y) - logloss(P[k], Y)
        bs = dif[rng.integers(0, len(dif), (20000, len(dif)))].mean(axis=1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print("  %-38s gain %+.5f/game  95%% CI [%+.5f, %+.5f]  P(better) %.3f"
              % (lab, dif.mean(), lo, hi, (bs > 0).mean()))

    acc, base = ((P["wf"] >= .5) == (Y == 1)).mean(), Y.mean()
    se = (0.25 / len(Y)) ** 0.5
    print("\n  accuracy %.2f%% vs %.2f%% base = %+.2fpp, ~%.2f SE"
          % (100 * acc, 100 * base, 100 * (acc - base), (acc - base) / se))
    dif = logloss(P["base"], Y) - logloss(P["wf"], Y)
    bs = dif[rng.integers(0, len(dif), (20000, len(dif)))].mean(axis=1)
    print("\n  VERDICT: %s"
          % ("model beats the constant at 95 percent" if np.percentile(bs, 2.5) > 0
             else "NO demonstrable skill - the interval spans zero"))
    print("  NOTE: injMiss is dropped (no historical injury feed). This is a FLOOR\n"
          "        on the live model, which carries that feature.")


if __name__ == "__main__":
    main()
