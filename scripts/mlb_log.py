#!/usr/bin/env python3
"""Server-side MLB full-game prediction log, with closing-line value.

Same treatment the WNBA log gets: predictions are recorded by a scheduled job
rather than by whichever browser happened to be open, so the record survives
independently of anyone's localStorage.

  LOG    - every PREGAME game gets the full-game model's home win probability
           plus the de-vigged Kalshi KXMLBGAME price at that moment.
  CLOSE  - every run overwrites `closing` while a game is still pregame, so the
           last value written before first pitch IS the closing line.
  GRADE  - once final, the pick is settled from the real score.
  CLV    - implied probability of the model's side at log time vs at close.
           Positive means the market moved TOWARD the model after it committed.

Why CLV rather than win rate: at ~15 games a day a 5-point drop in accuracy
takes about 1,500 games (~16 weeks) to separate from noise, because outcomes are
mostly variance. CLV measures whether the market agreed with us afterwards, which
is a far quieter signal and reads in weeks.

The model here MUST match REAL_MODEL / realModelRawStats in index.html.
scripts/verify_mlb_parity.py diffs the two and the workflow fails on drift.
"""
import json, math, os, urllib.request
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT  = os.path.join(ROOT, "data", "mlb_predictions.json")
MLB  = "https://statsapi.mlb.com/api/v1"

# ── constants — keep identical to index.html REAL_MODEL ──────────────
COEF      = [0.33448, 0.83292, 0.12567, 1.90351, -0.08084, 0.07588]
INTERCEPT = 0.11408
KBB_LG    = 0.14138
KBB_WBF   = 250
SHRINK_K  = 20          # shrink(val, n, prior=0.5, k=20) in index.html
PYTH_EXP  = 1.83


def get(url, tries=3):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "mlb-log/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception as e:
            last = e
    raise last


def sigmoid(z):
    z = max(-30.0, min(30.0, z))
    return 1.0 / (1.0 + pow(2.718281828459045, -z))


def shrink(val, n, prior=0.5, k=SHRINK_K):
    return (val * n + prior * k) / (n + k)


def pythagorean(rs, ra):
    if not rs or not ra or rs <= 0 or ra <= 0:
        return 0.5
    return rs ** PYTH_EXP / (rs ** PYTH_EXP + ra ** PYTH_EXP)


def kbb_quality(kbb, bf):
    if kbb is None or bf is None or bf < 40:
        return None
    wc = min(1.0, bf / KBB_WBF)
    return wc * kbb + (1 - wc) * KBB_LG


def model_prob(h, a, hkbb, akbb):
    """Mirror of realModelRawStats() in index.html."""
    if not h or not a:
        return None
    hg, ag = h["wins"] + h["losses"], a["wins"] + a["losses"]
    hPs = shrink(pythagorean(h["rs"], h["ra"]), hg)
    aPs = shrink(pythagorean(a["rs"], a["ra"]), ag)
    hWs = shrink(h["wins"] / hg if hg else 0.5, hg)
    aWs = shrink(a["wins"] / ag if ag else 0.5, ag)
    hRd = (h["rs"] - h["ra"]) / hg if hg else 0.0
    aRd = (a["rs"] - a["ra"]) / ag if ag else 0.0
    sp_known = 1 if (hkbb is not None and akbb is not None) else 0
    kbb_diff = (hkbb - akbb) if sp_known else 0.0
    f = [hPs - aPs, hWs - aWs, hRd - aRd, kbb_diff, sp_known, 1.0]
    return sigmoid(INTERCEPT + sum(c * x for c, x in zip(COEF, f)))


def team_abbrs():
    return {t["id"]: t["abbreviation"] for t in get(f"{MLB}/teams?sportId=1")["teams"]}


def standings_asof(date_str, ids):
    """Records through the END of date_str. `standings?date=D` INCLUDES games
    played on D, so callers must pass the day BEFORE the game to stay
    point-in-time (verified: 08-01 -> 3,334 team-games, 08-02 -> 3,364)."""
    d = get(f"{MLB}/standings?leagueId=103,104&season={date_str[:4]}&date={date_str}")
    out = {}
    for rec in d.get("records", []):
        for t in rec.get("teamRecords", []):
            ab = ids.get(t["team"]["id"])
            if ab:
                out[ab] = {"wins": t["wins"], "losses": t["losses"],
                           "rs": t.get("runsScored") or 0, "ra": t.get("runsAllowed") or 0}
    return out


def load_starter_kbb(pids, season):
    out = {}
    ids = [str(p) for p in dict.fromkeys([p for p in pids if p])]
    for i in range(0, len(ids), 100):
        chunk = ",".join(ids[i:i + 100])
        try:
            pj = get(f"{MLB}/people?personIds={chunk}"
                     f"&hydrate=stats(group=[pitching],type=[season],season={season})")
        except Exception as e:
            print(f"  ! kbb chunk: {e}")
            continue
        for p in pj.get("people", []):
            k = bb = bf = 0.0
            have = False
            for st in p.get("stats", []):
                for sp in st.get("splits", []):
                    s = sp.get("stat", {})
                    k += float(s.get("strikeOuts") or 0)
                    bb += float(s.get("baseOnBalls") or 0)
                    bf += float(s.get("battersFaced") or 0)
                    have = True
            if have and bf > 0:
                q = kbb_quality((k - bb) / bf, bf)
                if q is not None:
                    out[p["id"]] = q
    return out


def schedule(date):
    d = get(f"{MLB}/schedule?sportId=1&date={date}&hydrate=probablePitcher,team")
    games = []
    for day in d.get("dates", []):
        for g in day.get("games", []):
            t = g.get("teams", {})
            h, a = t.get("home", {}), t.get("away", {})
            state = (g.get("status", {}) or {}).get("abstractGameState", "")
            games.append({
                "gamePk": g.get("gamePk"), "date": date,
                "gameDate": g.get("gameDate"),
                "home": (h.get("team") or {}).get("id"),
                "away": (a.get("team") or {}).get("id"),
                "homeSpId": ((h.get("probablePitcher") or {}) or {}).get("id"),
                "awaySpId": ((a.get("probablePitcher") or {}) or {}).get("id"),
                "hs": h.get("score"), "as": a.get("score"),
                "isFinal": state == "Final", "isLive": state == "Live",
            })
    return games


def kalshi_game():
    """De-vigged Kalshi KXMLBGAME mid per event, keyed by sorted team pair.
    Reads the snapshot this same workflow writes a step earlier."""
    path = os.path.join(ROOT, "data", "kalshi_mlb.json")
    if not os.path.exists(path):
        return {}
    try:
        snap = json.load(open(path))
    except Exception:
        return {}
    by_event = {}
    for series, markets in (snap.get("markets") or {}).items():
        if "GAME" not in series.upper():
            continue
        for m in markets or []:
            by_event.setdefault(m.get("event_ticker"), []).append(m)

    def mid(m):
        try:
            b, a = float(m.get("yes_bid_dollars")), float(m.get("yes_ask_dollars"))
            if 0 <= b <= 1 and 0 < a <= 1 and b <= a:
                return (b + a) / 2
        except (TypeError, ValueError):
            pass
        return None

    out = {}
    for ticker, mkts in by_event.items():
        sides = {}
        for m in mkts:
            suf = (m.get("ticker") or "").rsplit("-", 1)[-1].upper()
            v = mid(m)
            if v is not None:
                sides[suf] = v
        if len(sides) != 2:
            continue
        (t1, p1), (t2, p2) = sorted(sides.items())
        tot = p1 + p2
        if tot <= 0:
            continue
        out["|".join(sorted([t1, t2]))] = {t1: p1 / tot, t2: p2 / tot}   # de-vigged
    return out


def main():
    store = {"entries": []}
    if os.path.exists(OUT):
        try:
            store = json.load(open(OUT))
        except Exception:
            pass
    entries = store.get("entries", [])
    by_id = {e["id"]: e for e in entries}

    ids = team_abbrs()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    days = [(datetime.now(timezone.utc) + timedelta(days=off)).strftime("%Y-%m-%d")
            for off in (-1, 0, 1)]
    games = [g for d in days for g in schedule(d)]
    kal = kalshi_game()
    season = today[:4]

    kbb = load_starter_kbb([g["homeSpId"] for g in games] +
                           [g["awaySpId"] for g in games], season)

    # point-in-time standings, one call per distinct prior day
    stand = {}
    for d in sorted({g["date"] for g in games}):
        prior = (datetime.strptime(d, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            stand[d] = standings_asof(prior, ids)
        except Exception as e:
            print(f"  ! standings {prior}: {e}")

    logged = closed = 0
    for g in games:
        hab, aab = ids.get(g["home"]), ids.get(g["away"])
        if not hab or not aab:
            continue
        st = stand.get(g["date"]) or {}
        p = model_prob(st.get(hab), st.get(aab), kbb.get(g["homeSpId"]), kbb.get(g["awaySpId"]))
        if p is None:
            continue
        key = "|".join(sorted([hab, aab]))
        mkt = kal.get(key)
        mkt_home = mkt.get(hab) if mkt else None
        gid = str(g["gamePk"])
        pregame = not g["isFinal"] and not g["isLive"]

        e = by_id.get(gid)
        if e is None and pregame:
            e = {"id": gid, "date": g["date"], "home": hab, "away": aab,
                 "gameDate": g["gameDate"], "model": {"home": round(p, 4)},
                 "kalshi": {"home": round(mkt_home, 4)} if mkt_home is not None else None,
                 "logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "result": None}
            entries.append(e); by_id[gid] = e; logged += 1
        if e is None:
            continue

        # closing line: overwrite while still pregame, so the last write sticks
        if pregame and mkt_home is not None:
            e["closing"] = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "kalshi": {"home": round(mkt_home, 4)}}
            closed += 1

        if g["isFinal"] and e.get("result") is None and g["hs"] is not None:
            if g["hs"] != g["as"]:
                e["result"] = {"homeWon": 1 if g["hs"] > g["as"] else 0,
                               "hs": g["hs"], "as": g["as"],
                               "graded_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    # ── CLV ──
    # For each entry, compare the implied probability of the MODEL'S SIDE at log
    # time vs at close. Positive means the market moved toward us after we
    # committed — the read that survives even when the results don't.
    # RAW CLV IS NOT A TEST OF THE MODEL. Prices drift toward the favourite
    # between log time and close, so any model that mostly picks favourites
    # collects that drift for free. Measured on 159 backfilled games: raw CLV
    # +0.52pp at z=3.09, but backing whatever side the market ALREADY favoured
    # earned +0.49pp on the same games, and the paired difference was +0.03pp
    # (z=0.23). The control is therefore computed alongside, and the PAIRED
    # difference is the number that answers the question.
    clv, ctrl = [], []
    for e in entries:
        c, k0 = e.get("closing"), e.get("kalshi")
        if not c or not k0 or not c.get("kalshi"):
            continue
        # On the first run of a slate the closing capture and the log-time
        # capture are the SAME snapshot, so every delta is exactly 0 and the
        # beat-rate reads 0% — which looks like a failing model when it only
        # means no movement has been observed yet. Require a later capture.
        if c.get("at") and e.get("logged_at") and c["at"] <= e["logged_at"]:
            continue
        side_home = e["model"]["home"] >= 0.5
        p0 = k0["home"] if side_home else 1 - k0["home"]
        p1 = c["kalshi"]["home"] if side_home else 1 - c["kalshi"]["home"]
        clv.append(p1 - p0)
        # control: the side the market itself favoured when we logged
        fav_home = k0["home"] >= 0.5
        f0 = k0["home"] if fav_home else 1 - k0["home"]
        f1 = c["kalshi"]["home"] if fav_home else 1 - c["kalshi"]["home"]
        ctrl.append(f1 - f0)
    if clv:
        avg = sum(clv) / len(clv)
        # A game where the price never moved is a push, not a loss. Counting it
        # in the denominator drags the beat rate toward zero and makes a quiet
        # market look like a failing model, so the rate is over games that
        # actually moved and `moved` is published alongside it.
        movers = [d for d in clv if abs(d) > 1e-9]
        beat = sum(1 for d in movers if d > 0)
        cavg = sum(ctrl) / len(ctrl)
        diff = [x - y for x, y in zip(clv, ctrl)]
        dm = sum(diff) / len(diff)
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
                  f"{cavg*100:+.2f}c -> incremental {dm*100:+.3f}c "
                  f"(z={dm/dse:+.2f})" if dse else
                  f"  CLV: {len(clv)} games, raw {avg*100:+.2f}c")
        else:
            print(f"  CLV: {len(clv)} games logged, none have moved yet")
    else:
        store.pop("clv", None)

    # ── straight accuracy, for context next to CLV ──
    hit = tot = 0
    for e in entries:
        r = e.get("result")
        if not r:
            continue
        tot += 1
        if (e["model"]["home"] >= 0.5) == (r["homeWon"] == 1):
            hit += 1
    if tot:
        store["accuracy"] = {"n": tot, "hit": hit, "rate": round(hit / tot, 4)}

    entries.sort(key=lambda e: (e["date"], e["id"]))
    store["entries"] = entries[-1200:]
    store["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(store, open(OUT, "w"), separators=(",", ":"))
    done = sum(1 for e in entries if e.get("result"))
    print(f"logged {logged} new, closing refreshed {closed}, "
          f"{len(entries)} entries, {done} graded -> {OUT}")


if __name__ == "__main__":
    main()
