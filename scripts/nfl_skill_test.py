#!/usr/bin/env python3
"""Does the NFL model have skill against a CONSTANT? Companion to
scripts/mlb_skill_test.py, which asked the same question of MLB and got no.

WHY ASK IT SEPARATELY FROM THE MARKET TEST. Beating a sharp price is hard and a
model can fail that while still being useful. Failing to beat "the home team
wins 53.9% of the time" is a different and much worse result: it means the
features carry no game-level signal and no shrinkage toward the price rescues
them. MLB failed this test (analysis/mlb_no_skill.md). This runs the same
protocol on football so the two are comparable.

METHOD. Walk-forward by DATE over 2020-2025: for each slate, fit ratings on
every game strictly before it and predict that slate. Eight seasons are fetched
(2018+) so the test period starts with a warm model; HALF_LIFE_DAYS=220 decays a
three-season-old game to ~3% weight, so fitting on deeper history is nearly the
shipped configuration rather than a different model.

THE SCALE IS THE SUBTLE PART. p = sigmoid(margin / scale), and fitting `scale`
on games the RATINGS have already seen answers with one far too small: in-sample
margins separate teams better than reality does. So the scale here is fitted the
way nfl_model.fit_margin_scale does it — hold out the most recent prior season,
fit ratings WITHOUT it, score margins ON it, fit the scale there. The in-sample
version is reported alongside purely to show the size of that bias.

Usage:  python scripts/nfl_skill_test.py [--cache PATH] [--from 2020]
"""
import argparse, importlib.util, json, math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_ns = importlib.util.spec_from_file_location("nm", os.path.join(HERE, "nfl_model.py"))
NM = importlib.util.module_from_spec(_ns); _ns.loader.exec_module(NM)
_cs = importlib.util.spec_from_file_location("cal", os.path.join(HERE, "calibration.py"))
C = importlib.util.module_from_spec(_cs); _cs.loader.exec_module(C)

FETCH_FROM = 2018
PRE_FIX_SCALE = 16.14      # what data/nfl_model.json shipped BEFORE 2026-08-26.
                           # Kept as the comparison baseline: this test is what
                           # condemned it. nfl_model.py now fits 12.26.


def load_games(cache):
    if cache and os.path.exists(cache):
        return json.load(open(cache))
    NM.SEASONS = list(range(FETCH_FROM, 2026))
    g = NM.fetch_games()
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        json.dump(g, open(cache, "w"))
    return g


def fit_scale(margins, ys, lo=3.0, hi=25.0, iters=80):
    """Golden-section, vectorised. Same objective as NM.fit_margin_scale; that
    one loops in Python and is far too slow to call once per slate."""
    m = np.asarray(margins, float); y = np.asarray(ys, float)
    def loss(s):
        p = np.clip(1 / (1 + np.exp(-np.clip(m / s, -30, 30))), 1e-12, 1 - 1e-12)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    gr = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - gr * (b - a), a + gr * (b - a)
    for _ in range(iters):
        if loss(c) < loss(d):
            b = d
        else:
            a = c
        c, d = b - gr * (b - a), a + gr * (b - a)
    return (a + b) / 2


def margins_for(model, rows):
    mg, ys = [], []
    for g in rows:
        if g["hs"] == g["as"]:
            continue                      # a tie carries no label
        ph, pa = NM.predict_points(model, g["home"], g["away"])
        mg.append(ph - pa); ys.append(1 if g["hs"] > g["as"] else 0)
    return mg, ys


def walk_forward(games, test_from):
    games = sorted(games, key=lambda g: g["date"])
    dates = sorted({g["date"][:10] for g in games if g["season"] >= test_from})
    P = {k: [] for k in ("base", "hfa_only", "oos", "insample", "shipped")}
    _scale_cache = {}
    Y, MARG, ACT, SC = [], [], [], []
    for i, d in enumerate(dates):
        dn = NM.day_number(d + "T00:00Z")
        prior = [g for g in games if NM.day_number(g["date"]) < dn]
        today = [g for g in games if g["date"][:10] == d and g["hs"] != g["as"]]
        if len(prior) < 400 or not today:
            continue
        model = NM.fit(prior, asof=dn)

        # HONEST scale, fitted ONCE PER SEASON and cached — not per slate.
        # Per-slate refitting with a one-season holdout is unstable: early in a
        # season the holdout ratings are stale and golden-section saturates at
        # its bounds (observed 25.00 and 4.51 on adjacent slates). Once per
        # season is also what nfl_model.py actually does when it ships a scale.
        season = next(g["season"] for g in today)
        if season not in _scale_cache:
            hold = season - 1
            fit_on = [g for g in games if g["season"] < hold]
            score = [g for g in games if g["season"] == hold]
            _scale_cache[season] = (fit_scale(*margins_for(NM.fit(fit_on), score))
                                    if fit_on and score else PRE_FIX_SCALE)
        sc_oos = _scale_cache[season]
        sc_in = fit_scale(*margins_for(model, prior))   # biased, for contrast

        base = float(np.mean([1 if g["hs"] > g["as"] else 0
                              for g in prior if g["hs"] != g["as"]]))
        for g in today:
            ph, pa = NM.predict_points(model, g["home"], g["away"])
            mg = ph - pa
            sig = lambda z: 1 / (1 + math.exp(-max(-30, min(30, z))))
            P["base"].append(base)
            P["hfa_only"].append(sig(model["hfa"] / sc_oos))
            P["oos"].append(sig(mg / sc_oos))
            P["insample"].append(sig(mg / sc_in))
            P["shipped"].append(sig(mg / PRE_FIX_SCALE))
            Y.append(1 if g["hs"] > g["as"] else 0)
            MARG.append(mg); ACT.append(g["hs"] - g["as"])
        SC.append((d, sc_oos, sc_in, model["hfa"]))
        if i % 60 == 0:
            print("  %s prior=%d  scale oos %.2f / in-sample %.2f  hfa %+.2f"
                  % (d, len(prior), sc_oos, sc_in, model["hfa"]), flush=True)
    return ({k: np.clip(np.array(v, float), 1e-6, 1 - 1e-6) for k, v in P.items()},
            np.array(Y, float), np.array(MARG), np.array(ACT), SC)


def logloss(p, y):
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(ROOT, "analysis", "nfl_games_deep.json"))
    ap.add_argument("--from", dest="test_from", type=int, default=2020)
    a = ap.parse_args()

    games = load_games(a.cache)
    print("loaded %d games, seasons %d-%d"
          % (len(games), min(g["season"] for g in games), max(g["season"] for g in games)))
    P, Y, MARG, ACT, SC = walk_forward(games, a.test_from)
    print("\nwalk-forward test games: %d (seasons %d+)   home rate %.4f\n"
          % (len(Y), a.test_from, Y.mean()))

    print("  %-32s %7s  %7s  %8s  %7s  %7s" % ("", "acc", "Brier", "logloss", "res", "rel"))
    for k, lab in (("base", "always home (prior rate)"),
                   ("hfa_only", "home-field only"),
                   ("oos", "MODEL, scale fitted OOS"),
                   ("insample", "MODEL, scale fitted in-sample"),
                   ("shipped", "MODEL, OLD scale %.2f (pre-fix)" % PRE_FIX_SCALE)):
        p = P[k]; d = C.brier_decomp(list(p), list(Y))
        print("  %-32s %6.2f%%  %.4f  %8.4f  %.4f  %.4f"
              % (lab, 100 * ((p >= .5) == (Y == 1)).mean(), ((p - Y) ** 2).mean(),
                 logloss(p, Y).mean(), d["resolution"], d["reliability"]))
    print("\n  mean absolute margin error %.2f pts" % np.abs(MARG - ACT).mean())

    rng = np.random.default_rng(11)
    print("\nPAIRED BOOTSTRAP on per-game log-loss (20k resamples)")
    for k, lab in (("oos", "MODEL(oos) vs always-home"),
                   ("hfa_only", "home-field only vs always-home")):
        dif = logloss(P["base"], Y) - logloss(P[k], Y)   # >0 = k better
        bs = dif[rng.integers(0, len(dif), (20000, len(dif)))].mean(axis=1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print("  %-32s gain %+.5f/game  95%% CI [%+.5f, %+.5f]  P(better) %.3f"
              % (lab, dif.mean(), lo, hi, (bs > 0).mean()))
    dif = logloss(P["shipped"], Y) - logloss(P["oos"], Y)
    bs = dif[rng.integers(0, len(dif), (20000, len(dif)))].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print("  %-32s gain %+.5f/game  95%% CI [%+.5f, %+.5f]  P(better) %.3f"
          % ("scale OOS vs OLD 16.14", dif.mean(), lo, hi, (bs > 0).mean()))

    acc, base = ((P["oos"] >= .5) == (Y == 1)).mean(), Y.mean()
    se = (0.25 / len(Y)) ** 0.5
    print("\n  accuracy %.2f%% vs %.2f%% base = %+.2fpp, ~%.2f SE"
          % (100 * acc, 100 * base, 100 * (acc - base), (acc - base) / se))
    recent = [s for s in SC if s[0] >= "2024"]
    if recent:
        print("  scale fitted OOS over 2024-25 slates: median %.2f (old shipped %.2f)"
              % (float(np.median([s[1] for s in recent])), PRE_FIX_SCALE))
    dif = logloss(P["base"], Y) - logloss(P["oos"], Y)
    bs = dif[rng.integers(0, len(dif), (20000, len(dif)))].mean(axis=1)
    print("\n  VERDICT: %s"
          % ("model beats the constant at 95 percent" if np.percentile(bs, 2.5) > 0
             else "NO demonstrable skill - the interval spans zero"))


if __name__ == "__main__":
    main()
