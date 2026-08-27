#!/usr/bin/env python3
"""NFL scoring model: team offense/defense ratings plus a quarter breakdown.

Built at Gabriel's request to give the NFL view what the WNBA view has — a
projected final AND a quarter-by-quarter table — rather than only the market's
number.

WHAT THIS IS AND IS NOT. Every model on this site has been measured against the
closing price and none has beaten it, so this ships as a PREDICTION, not a bet.
It is registered unproven (bet:false, w:0) exactly like MLB's was, it logs itself
forward, and it earns a place in Best Bets only if measurement later says so.
That is the same discipline the WNBA spread had to pass.

The design mirrors the WNBA model deliberately:
  - ratings fit on POINTS, not margin, because a quarter table needs both teams'
    scores and a margin model only gives their difference
  - recency weighting with prior-season carry, so a team that changed in the
    offseason is not judged on two-year-old form
  - walk-forward validation: fit on earlier seasons, score the later one, never
    on games the fit has seen

Quarter scoring uses the EMPIRICAL distribution of quarter points, exponentially
tilted to each team's expected rate. Three parametric families were tried first
and all of them failed the same way: a compound Poisson under-calls exactly-one-
touchdown quarters badly (17.7% modelled against 23.4% real) and over-calls
scoreless ones, and refitting the count as a negative binomial did not help —
the MLE dispersion came back at r=49.8, which is Poisson. The misfit is
structural, not over-dispersion: no i.i.d. touchdown-or-field-goal process puts
as much mass on a single touchdown as football does.

Tilting sidesteps the problem. p(x) is proportional to p_empirical(x)*exp(theta*x)
with theta solved so the mean matches the ratings, so the SHAPE is whatever
football actually does — 1- and 2-point quarters impossible, a spike at 7, ties
common — and only the level moves. At league average theta is 0 and the fit is
exact by construction.

Output: data/nfl_model.json
"""
import json, math, os, urllib.request
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "data", "nfl_model.json")
BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"

# WAS [2023, 2024, 2025]. The scale fit below holds out the most recent training
# season, so a 3-season list leaves _fit_on with exactly ONE season (2023) to
# build ratings from. Ratings off one season are weak, weak ratings separate
# teams poorly, and the optimiser answers that with a scale far too WIDE — 16.14
# where the converged value is 12.26. Measured on 1,693 walk-forward games
# (scripts/nfl_skill_test.py): 12.26 beats 16.14 by +0.00688 log-loss/game,
# 95% CI [+0.00350, +0.01017], and halves the calibration error (reliability
# 0.0031 vs 0.0053). Depth beyond this changes nothing — _fit_on of 4 seasons
# gives 12.28, of 6 gives 12.26 — because HALF_LIFE_DAYS already decays a
# 3-season-old game to ~3% weight. The list is long so _fit_on is never starved,
# not because old seasons matter.
SEASONS = list(range(2018, 2026))
TEST_SEASON = 2025          # held out for walk-forward scoring
HALF_LIFE_DAYS = 220.0      # ~1.3 seasons; a full prior year still counts, faintly
# Chosen by tuning on 2024 (fit 2023 only) and confirmed on an untouched 2025,
# not by feel. The old 12.0 over-shrank: ridge pulls every team rating toward
# zero while HFA and the intercept are deliberately UNregularised, so the
# systematic "home teams are stronger" signal migrated into the home-field term
# and was then applied to games where the home side is the weaker one. Measured
# on college, where it is worst: ridge 2 -> HFA +3.21 and a margin slope of
# 0.98; ridge 60 -> HFA +5.50 and a slope of 2.13. The slope IS the compression.
# The log-loss curve is flat from 1 to 8 here, so 3.0 sits at the optimum's
# shoulder rather than its tip — 32 teams and thin early-season data make the
# less aggressive end the safer place to stand.
RIDGE = 3.0

# All-star squads ESPN reports as teams. Never real franchises.
NON_TEAMS = {"AFC", "NFC"}


def get(url, tries=3):
    last = None
    for _ in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.load(r)
        except Exception as e:
            last = e
    raise last


def fetch_games():
    """Every completed regular-season and playoff game in SEASONS."""
    games = []
    for yr in SEASONS:
        for stype, weeks in ((2, range(1, 19)), (3, range(1, 6))):
            for wk in weeks:
                try:
                    d = get(f"{BASE}?dates={yr}&seasontype={stype}&week={wk}")
                except Exception as e:
                    print(f"  ! {yr} type{stype} wk{wk}: {e}")
                    continue
                for ev in d.get("events", []) or []:
                    c = (ev.get("competitions") or [{}])[0]
                    if ((c.get("status") or {}).get("type") or {}).get("name") != "STATUS_FINAL":
                        continue
                    home = next((t for t in c.get("competitors", []) if t.get("homeAway") == "home"), None)
                    away = next((t for t in c.get("competitors", []) if t.get("homeAway") == "away"), None)
                    if not home or not away:
                        continue
                    # The Pro Bowl arrives in seasontype 3 as a real final with
                    # AFC and NFC as the "franchises". It is not football: since
                    # 2023 it is a flag-football exhibition, and the seven in
                    # this range average 44.9 points a side against 23.0 for real
                    # games. Worse, it is played in February, so it is the most
                    # RECENT game in the set and recency weighting hands it the
                    # single largest weight of any row. It skews mu and the
                    # quarter tables, and registers two teams that never play again.
                    if {home["team"]["abbreviation"], away["team"]["abbreviation"]} & NON_TEAMS:
                        continue

                    def lines(t):
                        v = [x.get("value") for x in (t.get("linescores") or [])]
                        return [float(x) for x in v[:4]] if len(v) >= 4 else None

                    hl, al = lines(home), lines(away)
                    try:
                        hs, as_ = int(float(home["score"])), int(float(away["score"]))
                    except (KeyError, TypeError, ValueError):
                        continue
                    games.append({
                        "season": yr, "date": ev.get("date"),
                        "home": home["team"]["abbreviation"], "away": away["team"]["abbreviation"],
                        "hs": hs, "as": as_, "hl": hl, "al": al,
                    })
        print(f"  {yr}: {sum(1 for g in games if g['season']==yr)} final games")
    return games


def day_number(iso):
    """Days since 2000-01-01, for recency weighting. Cheap and monotonic."""
    y, m, d = int(iso[0:4]), int(iso[5:7]), int(iso[8:10])
    a = (14 - m) // 12
    yy = y + 4800 - a
    mm = m + 12 * a - 3
    jdn = d + (153 * mm + 2) // 5 + 365 * yy + yy // 4 - yy // 100 + yy // 400 - 32045
    return jdn - 2451545


def fit(games, asof=None):
    """Least-squares offense/defense ratings on POINTS, ridge-regularised.

        points_for = mu + off[attacker] + def[defender] + (hfa if attacker home)

    Fitting points rather than margin is what makes a quarter table possible:
    a margin model can tell you a team is favoured by 6 and still has nothing to
    say about whether the game is 27-21 or 13-7.
    """
    rows = [g for g in games if asof is None or day_number(g["date"]) < asof]
    teams = sorted({t for g in rows for t in (g["home"], g["away"])})
    ix = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    # columns: [off_0..off_{n-1}, def_0..def_{n-1}, hfa, mu]
    P = 2 * n + 2
    A, b, W = [], [], []
    latest = max(day_number(g["date"]) for g in rows) if rows else 0
    for g in rows:
        age = latest - day_number(g["date"])
        w = 0.5 ** (age / HALF_LIFE_DAYS)
        for atk, dfn, pts, is_home in ((g["home"], g["away"], g["hs"], 1),
                                       (g["away"], g["home"], g["as"], 0)):
            r = np.zeros(P)
            r[ix[atk]] = 1.0
            r[n + ix[dfn]] = 1.0
            r[2 * n] = is_home
            r[2 * n + 1] = 1.0
            A.append(r); b.append(pts); W.append(w)
    A = np.asarray(A); b = np.asarray(b); W = np.asarray(W)
    Aw = A * np.sqrt(W)[:, None]; bw = b * np.sqrt(W)
    # Ridge every parameter EXCEPT the intercept and home field, which are real
    # effects that should not be shrunk toward zero.
    R = np.eye(P) * RIDGE
    R[2 * n, 2 * n] = 0.0
    R[2 * n + 1, 2 * n + 1] = 0.0
    x = np.linalg.solve(Aw.T @ Aw + R, Aw.T @ bw)
    return {
        "teams": teams,
        "off": {t: float(x[ix[t]]) for t in teams},
        "def": {t: float(x[n + ix[t]]) for t in teams},
        "hfa": float(x[2 * n]),
        "mu": float(x[2 * n + 1]),
    }


def fit_margin_scale(margins, ys, lo=3.0, hi=25.0, iters=200):
    """The logistic scale minimising log loss: p = sigmoid(margin / scale).

    Golden-section on a 1-D objective. Fitted on seasons BEFORE the held-out one,
    so the walk-forward score below never sees a constant tuned on itself.
    """
    def loss(s):
        t = 0.0
        for m, y in zip(margins, ys):
            p = min(1 - 1e-12, max(1e-12, 1.0 / (1.0 + math.exp(-m / s))))
            t += -(y * math.log(p) + (1 - y) * math.log(1 - p))
        return t / len(margins)
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


def predict_points(m, home, away):
    """Expected points for each side. Unknown team falls back to league average."""
    o = m["off"]; d = m["def"]
    oh = o.get(home, 0.0); dh = d.get(home, 0.0)
    oa = o.get(away, 0.0); da = d.get(away, 0.0)
    return (m["mu"] + oh + da + m["hfa"], m["mu"] + oa + dh)


def quarter_shares(games):
    """Share of a team's points scored in each quarter, league-wide.

    Q2 and Q4 run hotter than Q1 and Q3 because both halves end with a clock
    that rewards aggression. Using a flat 25% would systematically under-call
    the second and fourth.
    """
    tot = [0.0, 0.0, 0.0, 0.0]; grand = 0.0
    for g in games:
        for ls in (g["hl"], g["al"]):
            if not ls:
                continue
            for i in range(4):
                tot[i] += ls[i]
            grand += sum(ls)
    return [t / grand for t in tot] if grand else [0.25] * 4


def empirical_quarter_pmf(games):
    """How often a team scores each exact point total in a quarter."""
    d = defaultdict(int)
    n = 0
    for g in games:
        for ls in (g["hl"], g["al"]):
            if not ls:
                continue
            for q in ls:
                d[int(q)] += 1
                n += 1
    return {int(k): v / n for k, v in sorted(d.items())} if n else {}


def tilt(base, target):
    """Move a pmf's mean to `target` while keeping its shape.

    p(x) proportional to base(x)*exp(theta*x). Bisection on theta because the
    mean is monotone in it. theta=0 leaves the empirical distribution untouched,
    which is what a league-average offence should get.
    """
    lo, hi = -3.0, 3.0
    for _ in range(200):
        th = (lo + hi) / 2
        w = {k: v * math.exp(th * k) for k, v in base.items()}
        s = sum(w.values())
        if (sum(k * v for k, v in w.items()) / s) < target:
            lo = th
        else:
            hi = th
    th = (lo + hi) / 2
    w = {k: v * math.exp(th * k) for k, v in base.items()}
    s = sum(w.values())
    return {k: v / s for k, v in w.items()}


def main():
    print("fetching…")
    games = fetch_games()
    print(f"{len(games)} final games, {sum(1 for g in games if g['hl'])} with quarter linescores")

    shares = quarter_shares(games)
    qpmf = empirical_quarter_pmf(games)
    mean_q = sum(k * v for k, v in qpmf.items())
    print(f"quarter shares Q1-Q4: {['%.3f'%x for x in shares]}")
    print(f"quarter pmf over {len(qpmf)} distinct totals, mean {mean_q:.2f} pts")

    # Does the tilted pmf reproduce real quarter outcomes? This is the check the
    # parametric versions failed.
    # Per-GAME, not a league-average matchup. Averaging one generic pairing and
    # comparing it to real games is wrong wherever home teams are systematically
    # stronger than away teams — which is exactly college football, where big
    # programs buy home games. That mistake read as a 3.9pp model error.
    tie = hw = 0.0
    ngm = 0
    _m0 = fit(games)
    for g in games:
        if not g["hl"] or not g["al"]:
            continue
        gh, ga = predict_points(_m0, g["home"], g["away"])
        ngm += 1
        for i in range(4):
            ph = tilt(qpmf, max(gh * shares[i], 0.05))
            pa = tilt(qpmf, max(ga * shares[i], 0.05))
            for x, px in ph.items():
                for y, py in pa.items():
                    if x == y:
                        tie += px * py
                    elif x > y:
                        hw += px * py
    tie /= (4 * ngm); hw /= (4 * ngm)
    et = eh = en = 0
    for g in games:
        if not g["hl"] or not g["al"]:
            continue
        for i in range(4):
            en += 1
            if g["hl"][i] == g["al"][i]:
                et += 1
            elif g["hl"][i] > g["al"][i]:
                eh += 1
    print(f"quarter tie  modelled {tie*100:.1f}% vs actual {100*et/en:.1f}%")
    print(f"quarter home modelled {hw*100:.1f}% vs actual {100*eh/en:.1f}%")

    # ── walk-forward: fit only on what came before, score the held-out season ──
    test = sorted([g for g in games if g["season"] == TEST_SEASON], key=lambda g: g["date"])

    # FIT the probability scale rather than picking one, and fit it OUT OF
    # SAMPLE. Fitting on predictions the model has already seen gives margins
    # that are far better separated than reality, so the optimiser answers with a
    # scale that is much too small and the whole model turns overconfident: on
    # NFL that produced 7.38 where the honest value is near 16, and the expected
    # calibration error went from 5.97pp to 9.99pp with resolution collapsing
    # from 0.0171 to 0.0058. Hold out the most recent training season, predict
    # it from the ones before, and fit the scale on THAT.
    pre = [g for g in games if g["season"] < TEST_SEASON]
    _hold = max(g["season"] for g in pre)
    _fit_on = [g for g in pre if g["season"] < _hold]
    _score = [g for g in pre if g["season"] == _hold]
    # _fit_on must be about as deep as the FINAL fit, or the scale is calibrated
    # for ratings weaker than the ones that will actually use it. See SEASONS.
    _nseas = len({g["season"] for g in _fit_on})
    if _nseas < 3:
        print(f"  ! WARNING: scale is being fitted against ratings built from only "
              f"{_nseas} season(s); it will come out too wide. Widen SEASONS.")
    _m = fit(_fit_on if _fit_on else pre)
    _mar, _ys = [], []
    for g in (_score if _fit_on else pre):
        if g["hs"] == g["as"]:
            continue
        _ph, _pa = predict_points(_m, g["home"], g["away"])
        _mar.append(_ph - _pa)
        _ys.append(1 if g["hs"] > g["as"] else 0)
    scale = fit_margin_scale(_mar, _ys)
    print(f"fitted margin scale {scale:.2f}  (out-of-sample on {_hold}, n={len(_mar)})")

    hit = n = 0
    ae = 0.0
    brier = 0.0
    for g in test:
        m = fit(games, asof=day_number(g["date"]))
        ph, pa = predict_points(m, g["home"], g["away"])
        margin = ph - pa
        # P(home win) from the margin, using the SD of NFL results about the spread
        p = 1.0 / (1.0 + math.exp(-margin / scale))
        won = 1 if g["hs"] > g["as"] else 0
        if g["hs"] != g["as"]:
            hit += (p >= 0.5) == (won == 1); n += 1
            brier += (p - won) ** 2
        ae += abs(margin - (g["hs"] - g["as"]))
    print(f"\nWALK-FORWARD on {TEST_SEASON}: {hit}/{n} = {100*hit/n:.1f}% straight up")
    print(f"  mean absolute margin error {ae/len(test):.2f} pts   Brier {brier/n:.4f}")

    full = fit(games)
    order = sorted(full["teams"], key=lambda t: -(full["off"][t] - full["def"][t]))
    print(f"\nhome field {full['hfa']:+.2f} pts   league mean {full['mu']:.2f} pts/team")
    print("strongest:", ", ".join(f"{t} {full['off'][t]-full['def'][t]:+.1f}" for t in order[:5]))
    print("weakest:  ", ", ".join(f"{t} {full['off'][t]-full['def'][t]:+.1f}" for t in order[-5:]))

    out = {
        "off": full["off"], "def": full["def"], "hfa": full["hfa"], "mu": full["mu"],
        "quarterShares": shares, "quarterPmf": qpmf, "marginScale": round(scale, 3),
        "trainedOn": len(games), "seasons": SEASONS,
        "walkForward": {"season": TEST_SEASON, "n": n, "hit": hit,
                        "rate": hit / n if n else None,
                        "mae": ae / len(test) if test else None,
                        "brier": brier / n if n else None},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), separators=(",", ":"))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
