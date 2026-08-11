"""Server-side WNBA prediction log.

Runs on a schedule from GitHub Actions so the track record no longer depends
on someone having a browser tab open before tip-off. Two jobs:

  1. LOG   — for every game that is still pregame, compute the model
             prediction (plus the Vegas / Kalshi prices at that moment) and
             append it to data/wnba_predictions.json if not already there.
  2. GRADE — for every logged entry whose game is now final, fill in the
             result from the box score.

The model here MUST stay in sync with the FIT block in index.html /
wnba_today.html. scripts/verify_wnba_parity.py compares the two and the
workflow fails if they diverge.
"""
import json, os, math, urllib.request, urllib.error
from datetime import datetime, timedelta, timezone

HERE   = os.path.dirname(os.path.abspath(__file__))
ROOT   = os.path.dirname(HERE)
OUT    = os.path.join(ROOT, "data", "wnba_predictions.json")
ESPN   = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba"
CORE   = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/wnba"

# ── model constants — keep identical to FIT in the HTML ──────────────
FIT = {
    "halfLife": 25, "priorW0": 4, "priorR": 0.6, "shrinkK": 8, "lgFallback": 83.0,
    "margin": {"eff": 1.0993, "homeEdge": 1.855, "b2b": -0.956, "injMiss": -0.1793, "sd": 12.33},
    "total":  {"eff": 1.1584, "intercept": -24.42, "b2b": 0.645, "injMiss": -0.0633},
    "winProbK": 0.1461,
    "spreadSd": 12.39,
    "quarters": [
        {"shareH": 0.2573, "shareA": 0.2550, "slope": 0.2510, "intercept":  0.200, "sd": 7.57, "tieBand": 0.477},
        {"shareH": 0.2504, "shareA": 0.2477, "slope": 0.3228, "intercept":  0.078, "sd": 7.11, "tieBand": 0.539},
        {"shareH": 0.2492, "shareA": 0.2548, "slope": 0.2919, "intercept": -0.543, "sd": 7.07, "tieBand": 0.479},
        {"shareH": 0.2431, "shareA": 0.2424, "slope": 0.1343, "intercept":  0.265, "sd": 7.11, "tieBand": 0.477},
    ],
    "priors2025": {
        "WSH": [76.07, 82.01], "ATL": [83.54, 75.99], "DAL": [81.55, 88.44], "MIN": [84.64, 77.11],
        "GS": [76.8, 76.41], "LA": [85.4, 87.95], "NY": [81.14, 79.83], "LV": [85.66, 80.5],
        "IND": [83.2, 80.85], "CHI": [75.42, 85.43], "PHX": [82.07, 80.68], "SEA": [81.49, 80.64],
        "CON": [76.27, 85.08],
    },
}
TEAMS = {
    "ATL": ("20", "Atlanta Dream"), "CHI": ("19", "Chicago Sky"), "CON": ("18", "Connecticut Sun"),
    "DAL": ("3", "Dallas Wings"), "GS": ("129689", "Golden State Valkyries"), "IND": ("5", "Indiana Fever"),
    "LV": ("17", "Las Vegas Aces"), "LA": ("6", "Los Angeles Sparks"), "MIN": ("8", "Minnesota Lynx"),
    "NY": ("9", "New York Liberty"), "PHX": ("11", "Phoenix Mercury"), "POR": ("132052", "Portland Fire"),
    "SEA": ("14", "Seattle Storm"), "TOR": ("131935", "Toronto Tempo"), "WSH": ("16", "Washington Mystics"),
}
NAME_TO_ABBR = {v[1]: k for k, v in TEAMS.items()}


def get(url, timeout=30):
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def sigmoid(z): return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
def norm_cdf(x, mu, sd): return 0.5 * (1 + math.erf((x - mu) / (sd * math.sqrt(2))))
def clamp(x, lo, hi): return max(lo, min(hi, x))
def shrink(val, n, prior): return (val * n + prior * FIT["shrinkK"]) / (n + FIT["shrinkK"])


def us_game_date(iso):
    """US date of the tip — a 7pm CT game is already tomorrow in UTC."""
    t = datetime.fromisoformat(iso.replace("Z", "+00:00")) if iso else datetime.now(timezone.utc)
    return (t - timedelta(hours=5)).strftime("%Y-%m-%d")


def ml_to_raw(ml):
    if ml is None: return None
    ml = float(ml)
    return 100.0 / (ml + 100.0) if ml > 0 else abs(ml) / (abs(ml) + 100.0)


def parse_event(ev):
    comp = (ev.get("competitions") or [{}])[0]
    st = ((comp.get("status") or {}).get("type") or {}).get("name", "")
    home = away = None
    for c in comp.get("competitors", []):
        (home := c) if c.get("homeAway") == "home" else (away := c)
    if not home or not away: return None
    def lines(c): return [int(float(x.get("value") or 0)) for x in (c.get("linescores") or [])]
    def score(c):
        try: return int(float(c.get("score")))
        except Exception: return 0
    odds = spread = None
    for o in comp.get("odds", []):
        if spread is None and o.get("pointSpread"):
            def pick(s):
                c2 = (s or {}).get("close") or (s or {}).get("open")
                if not c2: return None
                try: return {"line": float(c2["line"]), "odds": float(str(c2["odds"]).replace("+", ""))}
                except Exception: return None
            sh, sa = pick(o["pointSpread"].get("home")), pick(o["pointSpread"].get("away"))
            if sh and sa: spread = {"homeLine": sh["line"], "homeOdds": sh["odds"],
                                    "awayLine": sa["line"], "awayOdds": sa["odds"]}
        if odds is None:
            hml = (o.get("homeTeamOdds") or {}).get("moneyLine")
            aml = (o.get("awayTeamOdds") or {}).get("moneyLine")
            if hml is None and o.get("moneyline"):
                def pk(s):
                    v = ((s or {}).get("close") or {}).get("odds") or ((s or {}).get("open") or {}).get("odds")
                    try: return float(str(v).replace("+", ""))
                    except Exception: return None
                hml, aml = pk(o["moneyline"].get("home")), pk(o["moneyline"].get("away"))
            if hml is not None and aml is not None:
                odds = {"homeML": hml, "awayML": aml}
    return {
        "id": ev["id"],
        "home": (home.get("team") or {}).get("abbreviation"),
        "away": (away.get("team") or {}).get("abbreviation"),
        "homeScore": score(home), "awayScore": score(away),
        "homeLines": lines(home), "awayLines": lines(away),
        "date": ev.get("date"),
        "isFinal": "FINAL" in st,
        "isLive": any(k in st for k in ("IN_PROGRESS", "LIVE", "HALFTIME", "END_PERIOD")),
        "vegas": odds, "spread": spread,
    }


def scoreboard(ymd=None):
    url = f"{ESPN}/scoreboard" + (f"?dates={ymd}" if ymd else "")
    return [g for g in (parse_event(e) for e in get(url).get("events", [])) if g]


def load_ratings():
    """Recency-weighted PF/PA from game logs, seeded with prior-season ratings."""
    lam = 0.5 ** (1.0 / FIT["halfLife"])
    stats = {}
    for abbr, (tid, _) in TEAMS.items():
        wpf = wpa = wn = 0.0
        pr = FIT["priors2025"].get(abbr)
        if pr:
            wpf = (FIT["priorR"] * pr[0] + (1 - FIT["priorR"]) * FIT["lgFallback"]) * FIT["priorW0"]
            wpa = (FIT["priorR"] * pr[1] + (1 - FIT["priorR"]) * FIT["lgFallback"]) * FIT["priorW0"]
            wn = float(FIT["priorW0"])
        done = []
        try:
            d = get(f"{ESPN}/teams/{tid}/schedule")
            for ev in d.get("events", []):
                stype = ev.get("seasonType")
                stype = stype.get("type") if isinstance(stype, dict) else stype
                if stype is not None and stype not in (2, 3): continue
                comp = (ev.get("competitions") or [{}])[0]
                me = next((x for x in comp.get("competitors", []) if str((x.get("team") or {}).get("id")) == str(tid)), None)
                opp = next((x for x in comp.get("competitors", []) if x is not me and x.get("team")), None)
                if not me or not opp: continue
                def sc(x):
                    s = x.get("score")
                    if s is None: return None
                    try: return float(s["value"]) if isinstance(s, dict) else float(s)
                    except Exception: return None
                my, op = sc(me), sc(opp)
                if my is None or op is None or (my == 0 and op == 0): continue
                done.append((ev.get("date"), my, op, me.get("winner") is True))
        except Exception as e:
            print(f"  ! schedule {abbr}: {e}")
        done.sort(key=lambda x: x[0] or "")
        wins = losses = 0
        for _, my, op, won in done:
            wpf = wpf * lam + my; wpa = wpa * lam + op; wn = wn * lam + 1
            wins += 1 if won else 0; losses += 0 if won else 1
        stats[abbr] = {"wins": wins, "losses": losses, "gp": len(done), "wN": wn,
                       "pf": wpf / wn if wn > 0 else FIT["lgFallback"],
                       "pa": wpa / wn if wn > 0 else FIT["lgFallback"]}
    played = [s for s in stats.values() if s["gp"] > 0]
    lg = sum(s["pf"] for s in played) / len(played) if played else FIT["lgFallback"]
    return stats, lg


def load_injuries(stats, teams_playing):
    """PPG of OUT regulars (>=50% of team games, >=3 GP, >=6 PPG), capped at 30."""
    out = {}
    try:
        data = get(f"{ESPN}/injuries")
    except Exception as e:
        print(f"  ! injuries: {e}"); return out
    year = datetime.now(timezone.utc).year
    for team in data.get("injuries", []):
        abbr = NAME_TO_ABBR.get(team.get("displayName"))
        if not abbr or abbr not in teams_playing: continue
        names = [ (i.get("athlete") or {}).get("displayName")
                  for i in team.get("injuries", []) if str(i.get("status", "")).lower() == "out" ]
        names = [n for n in names if n]
        if not names: continue
        try:
            roster = get(f"{ESPN}/teams/{TEAMS[abbr][0]}/roster")
        except Exception: continue
        ids = {a.get("displayName"): a.get("id") for a in roster.get("athletes", [])}
        team_gp = stats.get(abbr, {}).get("gp", 0)
        if team_gp < 4: continue
        miss, who = 0.0, []
        for nm in names:
            aid = ids.get(nm)
            if not aid: continue
            try:
                sd = get(f"{CORE}/seasons/{year}/types/2/athletes/{aid}/statistics")
            except Exception: continue
            ppg = gp = None
            for cat in ((sd.get("splits") or {}).get("categories") or []):
                for s in cat.get("stats", []):
                    if s.get("name") == "avgPoints": ppg = s.get("value")
                    if s.get("name") == "gamesPlayed" and cat.get("name") == "general": gp = s.get("value")
            if ppg is None or gp is None: continue
            if gp >= 3 and gp / team_gp >= 0.5 and ppg >= 6:
                miss += ppg; who.append({"name": nm, "ppg": round(ppg, 1)})
        if miss > 0:
            out[abbr] = {"miss": round(min(miss, 30.0), 2), "names": who}
    return out


def predict(home, away, stats, lg, b2b, inj):
    h, a = stats.get(home), stats.get(away)
    if not h or not a: return None
    hpf = shrink(h["pf"], h["wN"], lg); hpa = shrink(h["pa"], h["wN"], lg)
    apf = shrink(a["pf"], a["wN"], lg); apa = shrink(a["pa"], a["wN"], lg)
    eh0 = hpf * apa / lg; ea0 = apf * hpa / lg
    bh = 1 if home in b2b else 0; ba = 1 if away in b2b else 0
    mh = inj.get(home, {}).get("miss", 0.0); ma = inj.get(away, {}).get("miss", 0.0)
    T = FIT["total"]["eff"] * (eh0 + ea0) + FIT["total"]["intercept"] \
        + FIT["total"]["b2b"] * (bh + ba) + FIT["total"]["injMiss"] * (mh + ma)
    mu = FIT["margin"]["eff"] * (eh0 - ea0) + FIT["margin"]["homeEdge"] \
        + FIT["margin"]["b2b"] * (bh - ba) + FIT["margin"]["injMiss"] * (mh - ma)
    exp_h, exp_a = (T + mu) / 2, (T - mu) / 2
    p_home = clamp(sigmoid(FIT["winProbK"] * mu), 0.03, 0.97)
    qs, scores = [], []
    for q in FIT["quarters"]:
        qh, qa = exp_h * q["shareH"], exp_a * q["shareA"]
        qmu = q["slope"] * mu + q["intercept"]
        p_away_q = norm_cdf(-q["tieBand"], qmu, q["sd"])
        p_home_q = 1 - norm_cdf(q["tieBand"], qmu, q["sd"])
        qs.append({"h": round(p_home_q, 4), "t": round(clamp(1 - p_home_q - p_away_q, 0, 1), 4),
                   "a": round(p_away_q, 4)})
        scores.append([round(qa, 1), round(qh, 1)])
    return {"game": round(p_home, 4), "mu": round(mu, 2),
            "expH": round(exp_h, 1), "expA": round(exp_a, 1), "q": qs, "scores": scores}


def kalshi_snapshot():
    """Reuse the market snapshot this workflow already produces."""
    path = os.path.join(ROOT, "data", "kalshi_wnba.json")
    if not os.path.exists(path): return {}, {}
    try: snap = json.load(open(path))
    except Exception: return {}, {}
    CITY = {"Atlanta": "ATL", "Chicago": "CHI", "Connecticut": "CON", "Dallas": "DAL",
            "Golden State": "GS", "Indiana": "IND", "Las Vegas": "LV", "Los Angeles": "LA",
            "Minnesota": "MIN", "New York": "NY", "Phoenix": "PHX", "Portland": "POR",
            "Seattle": "SEA", "Toronto": "TOR", "Washington": "WSH"}
    def mid(m):
        try:
            b, a = float(m.get("yes_bid_dollars")), float(m.get("yes_ask_dollars"))
            if b > 0 or a < 1: return (b + a) / 2
        except Exception: pass
        try: return float(m.get("last_price_dollars"))
        except Exception: return None
    def group(markets):
        by = {}
        for m in markets or []: by.setdefault(m.get("event_ticker"), []).append(m)
        out = {}
        for tick, ms in by.items():
            abbrs = {CITY[m["yes_sub_title"]] for m in ms if m.get("yes_sub_title") in CITY}
            if len(abbrs) == 2: out["|".join(sorted(abbrs))] = ms
        return out
    mk = snap.get("markets", {})
    return group(mk.get("KXWNBAGAME")), [group(mk.get(f"KXWNBA{q}QWINNER")) for q in (1, 2, 3, 4)]


def kalshi_probs(ms, home, away, three_way):
    if not ms: return None
    CITY = {"Atlanta": "ATL", "Chicago": "CHI", "Connecticut": "CON", "Dallas": "DAL",
            "Golden State": "GS", "Indiana": "IND", "Las Vegas": "LV", "Los Angeles": "LA",
            "Minnesota": "MIN", "New York": "NY", "Phoenix": "PHX", "Portland": "POR",
            "Seattle": "SEA", "Toronto": "TOR", "Washington": "WSH"}
    def mid(m):
        try:
            b, a = float(m.get("yes_bid_dollars")), float(m.get("yes_ask_dollars"))
            if b > 0 or a < 1: return (b + a) / 2
        except Exception: pass
        try: return float(m.get("last_price_dollars"))
        except Exception: return None
    h = a = t = None
    for m in ms:
        sub = m.get("yes_sub_title")
        if sub == "Tie": t = mid(m); continue
        ab = CITY.get(sub)
        if ab == home: h = mid(m)
        if ab == away: a = mid(m)
    if h is None or a is None: return None
    if three_way:
        t = 0.06 if t is None else t
        tot = h + a + t
        return None if tot <= 0 else {"h": round(h / tot, 4), "t": round(t / tot, 4), "a": round(a / tot, 4)}
    tot = h + a
    return None if tot <= 0 else round(h / tot, 4)


BETS = os.path.join(ROOT, "data", "wnba_bets.json")


def payout(odds):
    """Profit per 1 unit staked, from American odds."""
    return odds / 100.0 if odds > 0 else 100.0 / abs(odds)


def grade_bets(finals_by_id):
    """Grade the bets actually taken. Separate from the model's flags: the
    flags are what the model *suggested*, this ledger is what was really
    backed, and only the ledger answers whether it made money."""
    if not os.path.exists(BETS):
        return
    try:
        store = json.load(open(BETS))
    except Exception:
        return
    changed = False
    for b in store.get("bets", []):
        if b.get("result"):
            continue
        g = finals_by_id.get(str(b["gameId"]))
        if not g:
            continue
        hs, as_ = g["homeScore"], g["awayScore"]
        is_home = b["side"] == g["home"]
        margin = (hs - as_) if is_home else (as_ - hs)
        if b["type"] == "ML":
            outcome = "win" if margin > 0 else "loss"
        else:                                    # SPREAD
            adj = margin + float(b["line"])
            outcome = "push" if abs(adj) < 1e-9 else ("win" if adj > 0 else "loss")
        profit = 0.0 if outcome == "push" else (
            b["stakeUnits"] * payout(float(b["odds"])) if outcome == "win" else -b["stakeUnits"])
        b["result"] = {"outcome": outcome, "profitUnits": round(profit, 4),
                       "finalAway": as_, "finalHome": hs,
                       "graded_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        changed = True
        print(f"  bet {b['matchup']} {b['type']} {b['side']} -> {outcome.upper()} ({profit:+.2f}u)")
    if changed:
        settled = [b for b in store["bets"] if b.get("result")]
        won = sum(1 for b in settled if b["result"]["outcome"] == "win")
        lost = sum(1 for b in settled if b["result"]["outcome"] == "loss")
        pushed = sum(1 for b in settled if b["result"]["outcome"] == "push")
        pnl = sum(b["result"]["profitUnits"] for b in settled)
        staked = sum(b["stakeUnits"] for b in settled if b["result"]["outcome"] != "push")
        store["summary"] = {"won": won, "lost": lost, "pushed": pushed,
                            "profitUnits": round(pnl, 4), "stakedUnits": round(staked, 4),
                            "roi": round(pnl / staked, 4) if staked else None}
        store["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        json.dump(store, open(BETS, "w"), indent=1)
        print(f"  bet ledger: {won}-{lost}" + (f"-{pushed}push" if pushed else "") +
              f"  {pnl:+.2f}u  ROI {store['summary']['roi']:+.1%}" if staked else "")


def main():
    store = {"updated_at": None, "entries": []}
    if os.path.exists(OUT):
        try: store = json.load(open(OUT))
        except Exception: pass
    entries = store.get("entries", [])
    by_id = {e["id"]: e for e in entries}

    # Query explicit US dates rather than ESPN's default "today". The default
    # rolls over on ESPN's own clock, so an overnight run (e.g. 02:43 CT) still
    # served the PREVIOUS day's finished games and there was nothing pregame to
    # log — that silently produced an empty log. Asking for the current US date
    # (plus the next, for late tips) makes logging independent of when the
    # scheduled run actually fires, which matters because GitHub delays and
    # drops cron runs.
    now = datetime.now(timezone.utc)
    us_today = (now - timedelta(hours=5)).strftime("%Y%m%d")
    us_tomorrow = (now - timedelta(hours=5) + timedelta(days=1)).strftime("%Y%m%d")
    today = []
    seen_ids = set()
    for ymd in (us_today, us_tomorrow):
        try:
            for g in scoreboard(ymd):
                if g["id"] not in seen_ids:
                    seen_ids.add(g["id"]); today.append(g)
        except Exception as ex:
            print(f"  ! scoreboard {ymd}: {ex}")
    print(f"scoreboard {us_today}/{us_tomorrow}: {len(today)} games")
    pregame = [g for g in today if not g["isFinal"] and not g["isLive"] and g["id"] not in by_id]

    # ── LOG ──
    if pregame:
        stats, lg = load_ratings()
        y = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y%m%d")
        try: b2b = {t for g in scoreboard(y) for t in (g["home"], g["away"])}
        except Exception: b2b = set()
        playing = {t for g in pregame for t in (g["home"], g["away"])}
        inj = load_injuries(stats, playing)
        kg, kq = kalshi_snapshot()
        for g in pregame:
            m = predict(g["home"], g["away"], stats, lg, b2b, inj)
            if not m:
                print(f"  skip {g['away']}@{g['home']} (no ratings)"); continue
            key = "|".join(sorted([g["home"], g["away"]]))
            e = {
                "id": g["id"], "date": us_game_date(g["date"]),
                "home": g["home"], "away": g["away"],
                "model": m,
                "vegas": g["vegas"], "spread": g["spread"],
                "kalshi": {"game": kalshi_probs(kg.get(key), g["home"], g["away"], False),
                           "q": [kalshi_probs(q.get(key), g["home"], g["away"], True) for q in kq]},
                "injuries": {k: v for k, v in inj.items() if k in (g["home"], g["away"])},
                "logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "result": None,
            }
            entries.append(e); by_id[e["id"]] = e
            print(f"  logged {g['away']}@{g['home']}  P(home)={m['game']}  mu={m['mu']}")

    # ── CLOSING LINE ──
    # Every run overwrites `closing` for games that are still pregame, so the
    # last value written before tip IS the closing line. Closing-line value is
    # the fastest honest read on whether the model has edge: realized ROI needs
    # most of a season to separate a 3% edge from noise, CLV needs weeks,
    # because it measures whether the market moved toward us rather than
    # whether the ball bounced our way.
    still_pregame = [g for g in today if not g["isFinal"] and not g["isLive"]]
    if still_pregame:
        kg2, kq2 = kalshi_snapshot()
        for g in still_pregame:
            e = by_id.get(g["id"])
            if not e:
                continue
            key = "|".join(sorted([g["home"], g["away"]]))
            # Quarter prices were being fetched here and thrown away, which is why
            # `kalshi.q` is null on every logged game and the quarter market — the
            # thinnest one, and the likeliest place an edge survives — could never
            # be tested. Kalshi only opens quarter markets on a subset of games and
            # opens them LATER than the game market, so the pregame write usually
            # misses them; this closing pass is where they actually exist.
            qs = [kalshi_probs(q.get(key), g["home"], g["away"], True) for q in kq2]
            e["closing"] = {
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "vegas": g["vegas"], "spread": g["spread"],
                "kalshi": kalshi_probs(kg2.get(key), g["home"], g["away"], False),
                "kalshiQ": qs,
            }
            # Backfill the pregame slot too, so a game whose quarter markets opened
            # after the first write still has a usable pregame quarter price.
            if any(qs) and not any((e.get("kalshi") or {}).get("q") or []):
                e.setdefault("kalshi", {})["q"] = qs
        print(f"  closing-line refreshed for {len(still_pregame)} pregame games")

    # ── GRADE ──
    pending = [e for e in entries if not e.get("result")]
    if pending:
        dates = sorted({d for e in pending for d in (
            (datetime.strptime(e["date"], "%Y-%m-%d") + timedelta(days=off)).strftime("%Y%m%d")
            for off in (-1, 0, 1))})
        seen = {}
        for ymd in dates[-14:]:
            try:
                for g in scoreboard(ymd):
                    if g["isFinal"]: seen[g["id"]] = g
            except Exception as ex:
                print(f"  ! grade fetch {ymd}: {ex}")
        for e in pending:
            g = seen.get(e["id"])
            if not g: continue
            e["result"] = {
                "homeScore": g["homeScore"], "awayScore": g["awayScore"],
                "homeWon": 1 if g["homeScore"] > g["awayScore"] else 0,
                "qHome": g["homeLines"][:4], "qAway": g["awayLines"][:4],
                "graded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            ok = (e["model"]["game"] >= 0.5) == bool(e["result"]["homeWon"])
            print(f"  graded {e['away']}@{e['home']}  {g['awayScore']}-{g['homeScore']}  pick {'HIT' if ok else 'MISS'}")
        grade_bets({str(k): v for k, v in seen.items()})

    # ── CLV SUMMARY ──
    # For each entry where the model had a side, compare the implied
    # probability of that side at log time vs at close. Positive means the
    # market moved toward us — the signal that survives even when results don't.
    # See mlb_log.py: raw CLV is contaminated by drift toward the favourite, so
    # the favourite-backing control and the PAIRED difference are what matter.
    clv, ctrl = [], []
    for e in entries:
        c = e.get("closing")
        if not c or not e.get("vegas") or not c.get("vegas"):
            continue
        # On a slate's first run the closing capture and the log-time capture
        # are the SAME snapshot, so the delta is exactly 0 and it lands in the
        # beat-rate denominator as a loss. Require a genuinely later capture.
        if c.get("at") and e.get("logged_at") and c["at"] <= e["logged_at"]:
            continue
        p_model = e["model"]["game"]
        side_home = p_model >= 0.5
        def implied(v):
            h, a = ml_to_raw(v.get("homeML")), ml_to_raw(v.get("awayML"))
            if h is None or a is None or (h + a) <= 0:
                return None
            return (h / (h + a)) if side_home else (a / (h + a))
        p0, p1 = implied(e["vegas"]), implied(c["vegas"])
        if p0 is None or p1 is None:
            continue
        clv.append({"id": e["id"], "delta": p1 - p0})
        # control: whichever side the market favoured at log time
        h0, a0 = ml_to_raw(e["vegas"].get("homeML")), ml_to_raw(e["vegas"].get("awayML"))
        if h0 is not None and a0 is not None and (h0 + a0) > 0:
            fav_home = (h0 / (h0 + a0)) >= 0.5
            def imp2(v, home):
                hh, aa = ml_to_raw(v.get("homeML")), ml_to_raw(v.get("awayML"))
                if hh is None or aa is None or (hh + aa) <= 0:
                    return None
                return (hh / (hh + aa)) if home else (aa / (hh + aa))
            f0, f1 = imp2(e["vegas"], fav_home), imp2(c["vegas"], fav_home)
            ctrl.append({"id": e["id"], "delta": (f1 - f0) if (f0 is not None and f1 is not None) else 0.0})
        else:
            ctrl.append({"id": e["id"], "delta": 0.0})
    if clv:
        avg = sum(x["delta"] for x in clv) / len(clv)
        # A price that never moved is a PUSH, not a loss. Counting flat games in
        # the denominator drags the rate down and makes a quiet market look like
        # a failing model — measured here: 4 of 17 games were flat, which
        # published 35% when the rate over actual movers was 46%.
        movers = [x for x in clv if abs(x["delta"]) > 1e-9]
        beat = sum(1 for x in movers if x["delta"] > 0)
        cavg = sum(x["delta"] for x in ctrl) / len(ctrl) if ctrl else 0.0
        diff = [a["delta"] - b["delta"] for a, b in zip(clv, ctrl)]
        dm = sum(diff) / len(diff) if diff else 0.0
        dsd = math.sqrt(sum((x - dm) ** 2 for x in diff) / (len(diff) - 1)) if len(diff) > 1 else 0.0
        dse = dsd / math.sqrt(len(diff)) if diff else 0.0
        store["clv"] = {"n": len(clv), "moved": len(movers),
                        "avgCents": round(avg * 100, 2),
                        "ctrlCents": round(cavg * 100, 2),
                        "incrCents": round(dm * 100, 3),
                        "incrSeCents": round(dse * 100, 3),
                        "incrZ": round(dm / dse, 2) if dse else None,
                        "beatRate": round(beat / len(movers), 4) if movers else None}
        if movers:
            print(f"  CLV: {len(clv)} games, raw {avg*100:+.2f}c vs control "
                  f"{cavg*100:+.2f}c -> incremental {dm*100:+.3f}c"
                  + (f" (z={dm/dse:+.2f})" if dse else ""))
        else:
            print(f"  CLV: {len(clv)} games logged, none have moved yet")

    entries.sort(key=lambda e: (e["date"], e["id"]))
    store["entries"] = entries[-800:]
    store["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(store, open(OUT, "w"), separators=(",", ":"))
    done = sum(1 for e in entries if e.get("result"))
    print(f"total {len(entries)} entries, {done} graded -> {OUT}")


if __name__ == "__main__":
    main()
