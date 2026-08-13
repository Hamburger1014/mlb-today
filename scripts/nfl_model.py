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

SEASONS = [2023, 2024, 2025]
TEST_SEASON = 2025          # held out for walk-forward scoring
HALF_LIFE_DAYS = 220.0      # ~1.3 seasons; a full prior year still counts, faintly
RIDGE = 12.0                # shrinks thin samples toward league average


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
    tie = hw = 0.0
    m0 = fit(games)
    for i in range(4):
        ph = tilt(qpmf, (m0["mu"] + m0["hfa"]) * shares[i])
        pa = tilt(qpmf, m0["mu"] * shares[i])
        for x, px in ph.items():
            for y, py in pa.items():
                if x == y:
                    tie += px * py / 4
                elif x > y:
                    hw += px * py / 4
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
    hit = n = 0
    ae = 0.0
    brier = 0.0
    for g in test:
        m = fit(games, asof=day_number(g["date"]))
        ph, pa = predict_points(m, g["home"], g["away"])
        margin = ph - pa
        # P(home win) from the margin, using the SD of NFL results about the spread
        p = 1.0 / (1.0 + math.exp(-margin / 7.5))
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
        "quarterShares": shares, "quarterPmf": qpmf, "marginScale": 7.5,
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
