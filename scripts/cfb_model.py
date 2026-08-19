#!/usr/bin/env python3
"""College football scoring model — same machinery as the NFL one, retuned.

Imports the fit, the tilt and the quarter helpers from nfl_model rather than
copying them. Two copies of a fitter is precisely the drift this repo keeps
building verifiers to catch, and there is nothing sport-specific in the maths.

WHAT IS DIFFERENT ABOUT COLLEGE, and why the knobs move:

  ~135 FBS teams instead of 32, on a 12-game season. That is far less data per
  team, so RIDGE goes up — thin samples have to be pulled harder toward the
  league mean or a team with three blowouts looks like a juggernaut.

  Teams do not play a balanced schedule. An NFL rating is anchored by everyone
  playing 17 games against a roughly common pool; a college conference can be
  nearly disjoint from another. The ridge and the opponent-adjustment carry more
  weight here, and the ratings are worth less at the extremes.

  FCS opponents appear constantly — an FBS team schedules one most seasons — and
  they are not in the FBS pool. Given one or two games each, they would get wild
  individual ratings. They are pooled into a single OTHER team instead, which is
  the honest statement: we know they are weaker, we do not know their order.

  No preseason, so unlike the NFL view this one is live from week one.

Output: data/cfb_model.json
"""
import importlib.util
import json, math, os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "data", "cfb_model.json")

_spec = importlib.util.spec_from_file_location("nfl_model", os.path.join(HERE, "nfl_model.py"))
NM = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(NM)

BASE = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard"
SEASONS = [2023, 2024, 2025]
TEST_SEASON = 2025
MIN_GAMES = 8          # below this a team is pooled into OTHER
OTHER = "OTHER"


def fetch_games():
    """Every completed FBS game in SEASONS. groups=80 is the FBS filter; without
    it ESPN returns only a top-25 subset (20 events a week instead of 60)."""
    games = []
    for yr in SEASONS:
        n0 = len(games)
        for stype, weeks in ((2, range(1, 17)), (3, range(1, 3))):
            for wk in weeks:
                try:
                    d = NM.get(f"{BASE}?dates={yr}&seasontype={stype}&week={wk}&groups=80")
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
                        # Overtime adds periods; only the four quarters are modelled.
                        return [float(x) for x in v[:4]] if len(v) >= 4 else None

                    try:
                        hs, as_ = int(float(home["score"])), int(float(away["score"]))
                    except (KeyError, TypeError, ValueError):
                        continue
                    games.append({
                        "season": yr, "date": ev.get("date"),
                        "home": home["team"].get("abbreviation") or home["team"].get("id"),
                        "away": away["team"].get("abbreviation") or away["team"].get("id"),
                        "hs": hs, "as": as_, "hl": lines(home), "al": lines(away),
                    })
        print(f"  {yr}: {len(games)-n0} final games")
    return games


def pool_thin_teams(games):
    """Send anyone with too few games to a shared OTHER rating.

    Mostly FCS opponents. Giving each of them its own parameter off one game
    produces a rating that is pure noise and, worse, launders that noise into the
    FBS team that played them.
    """
    cnt = defaultdict(int)
    for g in games:
        cnt[g["home"]] += 1
        cnt[g["away"]] += 1
    thin = {t for t, n in cnt.items() if n < MIN_GAMES}
    for g in games:
        if g["home"] in thin:
            g["home"] = OTHER
        if g["away"] in thin:
            g["away"] = OTHER
    return games, len(thin)


def main():
    print("fetching…")
    games = fetch_games()
    games, nthin = pool_thin_teams(games)
    print(f"{len(games)} final games; pooled {nthin} thin teams into {OTHER}")
    with_lines = sum(1 for g in games if g["hl"])
    print(f"{with_lines} have quarter linescores")

    # WAS 30.0, on the reasoning that more teams on less data each should shrink
    # harder. That reasoning was wrong in a specific way: ridge shrinks team
    # ratings while HFA and the intercept are left UNregularised, so at 30 the
    # home-field term absorbed the fact that college home teams really are
    # stronger (big programs buy home games) and then applied it to every game
    # including the ones where the home side is weaker. Measured: ridge 2 gives
    # HFA +3.21 and a margin slope of 0.98, ridge 30 gives +4.92 and a slope of
    # 1.68, ridge 60 gives +5.50 and 2.13. Tuned on 2024 and confirmed on an
    # untouched 2025, where 5.0 beat 30.0 on every metric — log loss 0.5417 vs
    # 0.5679, ECE 4.06pp vs 4.55pp, accuracy 70.3% vs 68.7%.
    NM.RIDGE = 5.0
    NM.HALF_LIFE_DAYS = 200.0

    shares = NM.quarter_shares(games)
    qpmf = NM.empirical_quarter_pmf(games)
    mean_q = sum(k * v for k, v in qpmf.items())
    print(f"quarter shares Q1-Q4: {['%.3f'%x for x in shares]}")
    print(f"quarter pmf over {len(qpmf)} totals, mean {mean_q:.2f} pts")

    # Walk-forward on the held-out season. Refitting per game is the whole point:
    # a rating that has seen the result it is being scored on proves nothing.
    test = sorted([g for g in games if g["season"] == TEST_SEASON], key=lambda g: g["date"])
    hit = n = 0
    ae = 0.0
    brier = 0.0
    # Fitted OUT OF SAMPLE. See nfl_model.fit_margin_scale: fitting a scale on
    # predictions the model has already seen answers with one far too small,
    # because in-sample margins separate teams better than reality does.
    _pre = [g for g in games if g["season"] < TEST_SEASON]
    _hold = max(g["season"] for g in _pre)
    _fit_on = [g for g in _pre if g["season"] < _hold]
    _score = [g for g in _pre if g["season"] == _hold]
    _m0 = NM.fit(_fit_on if _fit_on else _pre)
    _mar, _ys = [], []
    for g in (_score if _fit_on else _pre):
        if g["hs"] == g["as"]:
            continue
        _ph, _pa = NM.predict_points(_m0, g["home"], g["away"])
        _mar.append(_ph - _pa)
        _ys.append(1 if g["hs"] > g["as"] else 0)
    SCALE = NM.fit_margin_scale(_mar, _ys)
    print(f"fitted margin scale {SCALE:.2f}  (out-of-sample on {_hold}, n={len(_mar)})")
    for g in test:
        m = NM.fit(games, asof=NM.day_number(g["date"]))
        ph, pa = NM.predict_points(m, g["home"], g["away"])
        margin = ph - pa
        p = 1.0 / (1.0 + math.exp(-margin / SCALE))
        if g["hs"] != g["as"]:
            won = 1 if g["hs"] > g["as"] else 0
            hit += (p >= 0.5) == (won == 1)
            n += 1
            brier += (p - won) ** 2
        ae += abs(margin - (g["hs"] - g["as"]))
    print(f"\nWALK-FORWARD on {TEST_SEASON}: {hit}/{n} = {100*hit/n:.1f}% straight up")
    print(f"  mean absolute margin error {ae/len(test):.2f} pts   Brier {brier/n:.4f}")

    full = NM.fit(games)
    # Per-GAME, not a league-average matchup. Averaging one generic pairing and
    # comparing it to real games is wrong wherever home teams are systematically
    # stronger than away teams — which is exactly college football, where big
    # programs buy home games. That mistake read as a 3.9pp model error.
    tie = hw = 0.0
    ngm = 0
    for g in games:
        if not g["hl"] or not g["al"]:
            continue
        gh, ga = NM.predict_points(full, g["home"], g["away"])
        ngm += 1
        for i in range(4):
            ph = NM.tilt(qpmf, max(gh * shares[i], 0.05))
            pa = NM.tilt(qpmf, max(ga * shares[i], 0.05))
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

    order = sorted(full["teams"], key=lambda t: -(full["off"][t] - full["def"][t]))
    print(f"\nhome field {full['hfa']:+.2f} pts   league mean {full['mu']:.2f} pts/team   {len(full['teams'])} teams")
    print("strongest:", ", ".join(f"{t} {full['off'][t]-full['def'][t]:+.1f}" for t in order[:6]))
    print("weakest:  ", ", ".join(f"{t} {full['off'][t]-full['def'][t]:+.1f}" for t in order[-4:]))

    out = {
        "off": full["off"], "def": full["def"], "hfa": full["hfa"], "mu": full["mu"],
        "quarterShares": shares, "quarterPmf": qpmf, "marginScale": round(SCALE, 3),
        "trainedOn": len(games), "seasons": SEASONS, "pooledInto": OTHER,
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
