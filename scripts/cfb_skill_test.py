#!/usr/bin/env python3
"""Does the CFB model have skill against a CONSTANT, and is its margin scale
starved the way NFL's was?

Third in the series: analysis/mlb_no_skill.md (no), analysis/nfl_skill.md (yes,
plus a scale 30% too wide). CFB is the one where the same structural bug was
SUSPECTED but its severity was unknown.

WHY CFB MIGHT BE FINE WHERE NFL WAS NOT. Both fit the scale by holding out the
most recent training season, and with a 3-season list both leave _fit_on with
exactly one season. But an NFL season is 286 games and a CFB season is ~911, so
CFB builds its holdout ratings on 3x the data. The starvation is the same in
shape and much milder in degree.

The walk-forward harness is IMPORTED from nfl_skill_test, not copied — see
cfb_model.py's own note on why two copies of a fitter is the drift this repo
keeps building verifiers to catch. Only the knobs differ: RIDGE 5.0,
HALF_LIFE_DAYS 200.0, and thin teams pooled into OTHER first.

Usage:  python scripts/cfb_skill_test.py [--from 2022]
"""
import argparse, importlib.util, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_cs = importlib.util.spec_from_file_location("cm", os.path.join(HERE, "cfb_model.py"))
CM = importlib.util.module_from_spec(_cs); _cs.loader.exec_module(CM)
_ss = importlib.util.spec_from_file_location("st", os.path.join(HERE, "nfl_skill_test.py"))
ST = importlib.util.module_from_spec(_ss); _ss.loader.exec_module(ST)
_cal = importlib.util.spec_from_file_location("cal", os.path.join(HERE, "calibration.py"))
C = importlib.util.module_from_spec(_cal); _cal.loader.exec_module(C)

NM = CM.NM
SHIPPED_SCALE = 11.709          # what data/cfb_model.json ships
CACHE = os.path.join(ROOT, "analysis", "cfb_games_deep.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="test_from", type=int, default=2022)
    ap.add_argument("--cache", default=CACHE)
    a = ap.parse_args()

    # cfb_model.main() sets these on the shared NM module; the harness reads them.
    NM.RIDGE = 5.0
    NM.HALF_LIFE_DAYS = 200.0

    if os.path.exists(a.cache):
        games = json.load(open(a.cache))
    else:
        CM.SEASONS = list(range(2018, 2026))
        games = CM.fetch_games()
        os.makedirs(os.path.dirname(a.cache), exist_ok=True)
        json.dump(games, open(a.cache, "w"))
    games, nthin = CM.pool_thin_teams([dict(g) for g in games])
    print("loaded %d games, seasons %d-%d; pooled %d thin teams into %s"
          % (len(games), min(g["season"] for g in games),
             max(g["season"] for g in games), nthin, CM.OTHER))

    # min_prior is higher than NFL's 400 because a CFB season is ~911 games:
    # 1200 is roughly "more than one full season seen", the same intent.
    P, Y, MARG, ACT, SC = ST.walk_forward(games, a.test_from,
                                          ref_scale=SHIPPED_SCALE, min_prior=1200)
    print("\nwalk-forward test games: %d (seasons %d+)   home rate %.4f\n"
          % (len(Y), a.test_from, Y.mean()))

    print("  %-34s %7s  %7s  %8s  %7s  %7s" % ("", "acc", "Brier", "logloss", "res", "rel"))
    for k, lab in (("base", "always home (prior rate)"),
                   ("hfa_only", "home-field only"),
                   ("oos", "MODEL, scale fitted OOS"),
                   ("insample", "MODEL, scale fitted in-sample"),
                   ("shipped", "MODEL, shipped scale %.3f" % SHIPPED_SCALE)):
        p = P[k]; d = C.brier_decomp(list(p), list(Y))
        print("  %-34s %6.2f%%  %.4f  %8.4f  %.4f  %.4f"
              % (lab, 100 * ((p >= .5) == (Y == 1)).mean(), ((p - Y) ** 2).mean(),
                 ST.logloss(p, Y).mean(), d["resolution"], d["reliability"]))
    print("\n  mean absolute margin error %.2f pts" % np.abs(MARG - ACT).mean())

    rng = np.random.default_rng(11)
    print("\nPAIRED BOOTSTRAP on per-game log-loss (20k resamples)")
    for lo_k, hi_k, lab in (("base", "oos", "MODEL(oos) vs always-home"),
                            ("base", "hfa_only", "home-field only vs always-home"),
                            ("shipped", "oos", "scale OOS vs shipped %.3f" % SHIPPED_SCALE)):
        dif = ST.logloss(P[lo_k], Y) - ST.logloss(P[hi_k], Y)
        bs = dif[rng.integers(0, len(dif), (20000, len(dif)))].mean(axis=1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print("  %-34s gain %+.5f/game  95%% CI [%+.5f, %+.5f]  P(better) %.3f"
              % (lab, dif.mean(), lo, hi, (bs > 0).mean()))

    acc, base = ((P["oos"] >= .5) == (Y == 1)).mean(), Y.mean()
    se = (0.25 / len(Y)) ** 0.5
    print("\n  accuracy %.2f%% vs %.2f%% base = %+.2fpp, ~%.2f SE"
          % (100 * acc, 100 * base, 100 * (acc - base), (acc - base) / se))
    recent = [s for s in SC if s[0] >= "2024"]
    if recent:
        print("  scale fitted OOS over 2024-25 slates: median %.2f (shipped %.3f)"
              % (float(np.median([s[1] for s in recent])), SHIPPED_SCALE))
    dif = ST.logloss(P["base"], Y) - ST.logloss(P["oos"], Y)
    bs = dif[rng.integers(0, len(dif), (20000, len(dif)))].mean(axis=1)
    print("\n  VERDICT: %s"
          % ("model beats the constant at 95 percent" if np.percentile(bs, 2.5) > 0
             else "NO demonstrable skill - the interval spans zero"))


if __name__ == "__main__":
    main()
